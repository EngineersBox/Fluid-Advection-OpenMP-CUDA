// OpenMP parallel 2D advection solver module
// template written for COMP4300/8300 Assignment 2, 2021
// template v1.0 14 Apr 

// ==== INCLUDES ====

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <complex.h>
#include <memory.h>

#include "serAdvect.h" // advection parameters

// ==== CONSTRUCT DEFINES ==== 

#define STATIC_INLINE __attribute__((always_inline)) static inline
#define mat2 double*
#define mat2_r mat2 restrict

#define for_rt(T, var, lower, upper) for (T var = (lower); (var) < (upper); (var)++)
#define for_rtc(T, var, lower, upper) for_rt(T, var, lower, (lower) + (upper))
#define for_r(var, lower, upper) for_rt(size_t, var, lower, upper)
#define for_rc(var, lower, upper) for_rt(size_t, var, lower, (lower) + (upper))
#define for_ru(var, lower, upper) for_rt(, var, lower, upper)
#define for_rcu(var, lower, upper) for_rt(, var, lower, (lower) + (upper))

// ==== BEHAVIOURAL DEFINES ====

#ifndef FFT_CONV_KERNEL
#define FFT_CONV_KERNEL 1
#endif

#if FFT_CONV_KERNEL == 1
#include <fftw3.h>
#include <stdbool.h>
#endif

#ifndef LOG_2D_EXCHANGES
#define LOG_2D_EXCHANGES 0
#endif

#ifndef SWAP_BUFFERS
#define SWAP_BUFFERS 1
#endif

#if SWAP_BUFFERS == 1
#define swap(a, b, T) do { T swap = a; a = b; b = swap; } while (0)
#endif

// ==== COMPILE-TIME SWITCHING ====

#define __INVOKE_ARGS(...) __VA_ARGS__
#define __IGNORE_ARGS(...) ({})

#if FFT_CONV_KERNEL == 1
#define _FFT_TD __INVOKE_ARGS
#define _FFT_FD __IGNORE_ARGS
#else
#define _FFT_TD __IGNORE_ARGS
#define _FFT_FD __INVOKE_ARGS
#endif

#if LOG_2D_EXCHANGES == 1
#define _EXCHANGE_TD __INVOKE_ARGS
#define _EXCHANGE_FD __IGNORE_ARGS
#else
#define _EXCHANGE_TD __IGNORE_ARGS
#define _EXCHANGE_FD __INVOKE_ARGS
#endif

#if SWAP_BUFFERS == 1
#define _SWAP_TD __INVOKE_ARGS
#define _SWAP_FD __IGNORE_ARGS
#else
#define _SWAP_TD __IGNORE_ARGS
#define _SWAP_FD __INVOKE_ARGS
#endif

#define _COMPILE_COND(FUNC, ...) _##FUNC##D(__VA_ARGS__)
#define COMPILE_COND_T(FUNC, ...) _COMPILE_COND(FUNC##_T, __VA_ARGS__)
#define COMPILE_COND_F(FUNC, ...) _COMPILE_COND(FUNC##_F, __VA_ARGS__)

// ==== PROGRAM ====

static int M, N, P, Q; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
	M = M_, N = N_; P = P_, Q = Q_;
	verbosity = verb;
} //initParParams()

STATIC_INLINE void omp1dUpdateBoundary(mat2 u, int ldu) {
	int i, j;
	#pragma omp for private(j)
	for_ru (j, 1, N + 1) { //top and bottom halo
		V(u, 0, j)   = V(u, M, j);
		V(u, M+1, j) = V(u, 1, j);
	}
	#pragma omp for private(i)
	for_ru (i, 0, M + 2) { //left and right sides of halo 
		V(u, i, 0) = V(u, i, N);
		V(u, i, N+1) = V(u, i, 1);
	}
} 

STATIC_INLINE void omp1dUpdateAdvectField(mat2_r u, int ldu, mat2_r v, int ldv) {
	int i, j;
	double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
	double cim1, ci0, cip1, cjm1, cj0, cjp1;
	N2Coeff(Ux, &cim1, &ci0, &cip1); N2Coeff(Uy, &cjm1, &cj0, &cjp1);
	#pragma omp for private(i)
	for_ru (i, 0, M)
		for_ru (j, 0, N) 
			V(v,i,j) =
				cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
				ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
				cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
} //omp1dUpdateAdvectField()  

STATIC_INLINE void omp1dCopyField(mat2_r v, int ldv, mat2_r u, int ldu) {
	int i, j;
	#pragma omp for private(i)
	for_ru (i, 0, M)
		for_ru (j, 0, N)
			V(u,i,j) = V(v,i,j);
} //omp1dCopyField()

// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, mat2 u, int ldu) {
	int ldv = N+2;
	mat2 v = calloc(ldv*(M+2), sizeof(*v));
	assert(v != NULL);
	for_r (r, 0, reps) {    
		#pragma omp parallel shared(u,ldu,v,ldv)
		{
			omp1dUpdateBoundary(u, ldu);
			omp1dUpdateAdvectField(&V(u,1,1), ldu, &V(v,1,1), ldv);
			omp1dCopyField(&V(v,1,1), ldv, &V(u,1,1), ldu);
		}
	} //for (r...)
	free(v);
} //omp1dAdvect()

/**
 * This avoids the use of if statements which cause warp/wavefront divergence.
 * The reason is that, since we are using SIMD parallelism, then when we have a condition that
 * only some threads meet, the warp/wavefront is split into two execution units using bit masks
 * for each thread that either satisfies or doesn't satisfy the condition. This is called
 * wrap/wavefront divergence. These split warps/wavefronts are executed sequentially, being treated
 * as if two different instructions are being processed.
 *
 * ==== Example with if statements ==== 
 *
 * CODE:
 * 1. int threadIdx = get_global_id(0); // Get current thread in 1D block of n threads
 * 2. int threadSpecificValue;
 * 3. if (threadIdx % 2 == 0) {
 * 4.     threadSpecificValue = func(12);
 * 5. } else {
 * 6.     threadSpecificValue = func(5);
 * 7. }
 * 8. char value = someOtherOp(threadSpecificValue);
 * 
 * Kernel:
 *  - Global work size (block): 1
 *  - Local work size (item): 10
 * 
 * +------+------------------+
 * | Time | Warp Thread Mask | <-- 0: Not active (NOP), 1: Active (INSN)
 * +------+------------------+
 * | 0    | 1111111111       | <-- [Line: 1] First line executes on all threads
 * | ======= DIVERGE ======= | <-- [Line: 3] We reach the condition that is specific to only some threads
 * | 1    | 0101010101       | <-- [Line: 4] The scheduler has prioritised a particular thread mask first, only the even threads will execute
 * | 2    | 1010101010       | <-- [Line: 6] Now the odd threads will execute (once previous insn has finished, causing NOPs)
 * | ====== CONVERGE ======= | <-- [Line: 7] Warp has converged again, next instruction is not thread id specific
 * | 3    | 1111111111       | <-- [Line: 8] Execution has return to normal, we no longer have thread specific execution
 * +------+------------------+
 * 
 * This is bad for parallelism as some of the threads in the warp/wavefront are just NOPs,
 * while others are executing. We can remove this issue altogether by using arithmetic operations
 * for conditions instead of actual comparisons.
 *
 * ==== Example with arithmetic operations ====
 *
 * CODE:
 * 1. int threadIdx = get_global_id(0); // Get current thread in 1D block of n threads
 * 2. int modi = threadIdx % 2;
 * 3. int threadSpecificValue = func((modi * 5) + ((1 - modi) * 12)); // Using the property of multiplying by zero on alternating threads to choose between values
 * 4. char value = someOtherOp(threadSpecificValue);
 * 
 * Kernel:
 *  - Global work size (block): 1
 *  - Local work size (item): 10
 * 
 * +------+------------------+
 * | Time | Warp Thread Mask | <-- 0: Not active (NOP), 1: Active (INSN)
 * +------+------------------+
 * | 0    | 1111111111       | <-- [Line: 1] First line executes on all threads
 * | 1    | 1111111111       | <-- [Line: 3] Our previously, thread specific line execution is now just an arithmetic operation not specific to threads
 * | 2    | 1111111111       | <-- [Line: 8] Execution has return to normal, we no longer have thread specific execution
 * +------+------------------+
 *
 * Using this simple trick, we have successfully returned out code back to being fully parallel
 * and ensured that it also takes less overall time to execute.
 */
#define modAlt(modi, trueValue, falseValue) (((modi) * (trueValue)) + ((1 - (modi)) * (falseValue)))

#define EXCHANGE_PARAMS size_t threadId, size_t i, size_t j, mat2 u, size_t ldu, size_t N_loc, size_t M_loc, size_t M0, size_t N0

void exchangeBlockCorner(EXCHANGE_PARAMS) {
	// Exchange single corner
	size_t horizontalCond = i == 0;
	size_t verticalCond = j == 0;
	COMPILE_COND_T(EXCHANGE,
		printf(
				"[C] [%zu] [%zu,%zu] => [%zu,%zu]\n",
				threadId,
				modAlt(verticalCond, M , 1),
				modAlt(horizontalCond, N, 1),
				modAlt(verticalCond, 0, M + 1),
				modAlt(horizontalCond, 0, N + 1)
		);
	);
	V(u, modAlt(verticalCond, 0, M + 1), modAlt(horizontalCond, 0, N + 1)) = V(u, modAlt(verticalCond, M, 1), modAlt(horizontalCond, N, 1));
	// Exchange top or bottom
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[C] [%zu] yDst: %zu, ySrc: %zu, x: [%zu,%zu)\n",
			threadId,
			yDst, ySrc,
			N0, N0 + N_loc
		);
	);
	for_rc (x, N0, N_loc) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
	// Exchange left or right
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[C] [%zu] xDst: %zu, xSrc: %zu, y: [%zu,%zu)\n",
			threadId,
			xDst, xSrc,
			M0, M0 + M_loc
		);
	);
	for_rc (y, M0, M_loc) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeBlockTopBottom(EXCHANGE_PARAMS) {
	// Exchange top or bottom
	size_t verticalCond = j == 0;
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[%s] [%zu] yDst: %zu, ySrc: %zu, x: [%zu,%zu)\n",
			verticalCond ? "T" : "B",
			threadId,
			yDst, ySrc,
			N0, N0 + N_loc
		);
	);
	for_rc (x, N0, N_loc) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
}

void exchangeBlockLeftRight(EXCHANGE_PARAMS) {
	// Exchange left or right
	size_t horizontalCond = i == 0;
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[%s] [%zu] xDst: %zu, xSrc: %zu, y: [%zu,%zu)\n",
			horizontalCond ? "L" : "R",
			threadId,
			xDst, xSrc,
			M0, M0 + M_loc
		);
	);
	for_rc (y, M0, M_loc) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeNoop(EXCHANGE_PARAMS) {}

typedef void(*ExchangeHandler)(EXCHANGE_PARAMS);

#define handlerAlt(modi, exchangeHandler) ((ExchangeHandler) modAlt((modi), (uintptr_t) &(exchangeHandler), (uintptr_t) &exchangeNoop))
#define cornerMod(i, j)    (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) == 0)
#define topBottomMod(i, j) (uintptr_t) ((i) % (P - 1) != 0 && (j) % (Q - 1) == 0)
#define leftRightMod(i, j) (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) != 0)

// ... using 2D parallelization
void omp2dAdvect(int reps, mat2 u, int ldu) {
	size_t i, j, r, ldv = N+2;
	mat2 v = calloc(ldv * (M + 2), sizeof(*v));
	assert(v != NULL);

	for_ru (r, 0, reps) {
		#pragma omp parallel for shared(u, ldu, v, ldv) private(i, j) collapse(2)
		for_ru (i, 0, P) {
			for_ru (j, 0, Q) {
				size_t threadId = omp_get_thread_num();
				size_t P0 = threadId / Q;
				size_t M0 = ((M / P) * P0) + 1;
				size_t M_loc = (P0 < P - 1) ? (M / P) : (M - M0);
				size_t Q0 = threadId % Q;
				size_t N0 = ((N / Q) * Q0) + 1;
				size_t N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);
				// Pointer arithmetic to avoid conditionals and warp divergence
				ExchangeHandler corner = handlerAlt(cornerMod(i, j), exchangeBlockCorner);
				corner(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				ExchangeHandler topBottom = handlerAlt(topBottomMod(i, j), exchangeBlockTopBottom);
				topBottom(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				ExchangeHandler leftRight = handlerAlt(leftRightMod(i, j), exchangeBlockLeftRight);
				leftRight(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				COMPILE_COND_T(EXCHANGE,
					printf(
						"[%zu] i = %zu, j = %zu, corner: %p, topBottom: %p, leftRight: %p\n",
						threadId,
						i, j,
						(void*) ((uintptr_t) (corner != exchangeNoop) * (uintptr_t) corner),
						(void*) ((uintptr_t) (topBottom != exchangeNoop) * (uintptr_t) topBottom),
						(void*) ((uintptr_t) (leftRight != exchangeNoop) * (uintptr_t) leftRight)
					);
				);
				updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv);
				COMPILE_COND_F(SWAP, copyField(M_loc, N_loc, &V(v, M0, N0), ldv, &V(u, M0, N0), ldu));
			}
		}
		COMPILE_COND_T(SWAP, swap(u, v, mat2));
	} //for (r...)
	COMPILE_COND_T(SWAP,
		if (reps % 2 != 0) {
			swap(u, v, double*);
		}
	);
	free(v);
} //omp2dAdvect()

#if FFT_CONV_KERNEL == 1
void fft_forward(mat2_r v, fftw_complex* restrict output_buffer, int n) {

}

void fft_backward(fftw_complex* restrict input_buffer, mat2_r output, int n) {

}

void convolution_fftw_2d(mat2_r real, mat2_r input, mat2_r result) {
	int n_formula;
	fftw_complex* a_complex;
	fftw_complex* input_complex;
	fftw_complex* odd_mults;
	fft_forward(real,a_complex,n_formula);
	bool is_initialised = false;
	int t;
	size_t i;
	while (t > 1) {
		if (t & 1) {
			if (!is_initialised) {
				memcpy(odd_mults, a_complex, n_formula * n_formula);
				is_initialised = true;
			} else {
				#pragma omp for private(i)
				for (i = 0; i < n_formula * n_formula; i++) {
					odd_mults[i] = odd_mults[i] * a_complex[i];
				}
			}
		}
		#pragma omp for private(i)
		for (i = 0; i < n_formula * n_formula; i++) {
			a_complex[i] = a_complex[i] * a_complex[i];
		}
		t /= 2;
	}
	if (is_initialised) {
		#pragma omp for private(i)
		for (i  = 0; i < n_formula * n_formula; i++) {
			a_complex[i] = a_complex[i] * odd_mults[i];
		}
	}

	fft_forward(input, input_complex, N);

	#pragma omp for private(i)
	for (i = 0; i < N * N; i++) {
		a_complex[i] = input_complex[i] * a_complex[i];
	}

	fft_backward(a_complex, result, N);
}

#endif

// ... extra optimization variant
void ompAdvectExtra(int r, mat2 u, int ldu) {
#if FFT_CONV_KERNEL == 1
	int timesteps = r;
	fftw_complex* a_complex = (fftw_complex*) fftw_alloc_complex(M * N);

	assert(a_complex != NULL);
	memset(a_complex, 0, sizeof(fftw_complex) * M * N);
	fftw_plan plan_u_f = fftw_plan_dft_r2c_2d(M, N, u, a_complex, FFTW_ESTIMATE);
	fftw_execute(plan_u_f);
	fftw_destroy_plan(plan_u_f);

	fftw_complex* tempSquaring = (fftw_complex*) fftw_alloc_complex(M_loc * N_loc);
	memset(tempSquaring, 0, M_loc * N_loc * sizeof(*tempSquaring));
	/*MPI_Scatter(*/
			/*a_complex, 1, matSegType,*/
			/*tempSquaring, 1, matSegType,*/
			/*0, commHandle*/
	/*);*/
	#pragma omp parallel
	{
		repeatedSquaring(timesteps, tempSquaring, M_loc * N_loc);
		/*MPI_Gather(*/
			/*tempSquaring, 1, matSegType,*/
			/*a_complex, 1, matSegType,*/
			/*0, commHandle*/
		/*);*/
		fftw_free(tempSquaring);

		fftw_complex* input_complex = (fftw_complex*) fftw_alloc_complex(M * N);
		fftw_plan plan_lwk_f = fftw_plan_dft_r2c_2d(M, N, laxWendroffKernel, input_complex, FFTW_ESTIMATE);
		fftw_execute(plan_lwk_f);
		fftw_destroy_plan(plan_lwk_f);

		tempSquaring = (fftw_complex*) fftw_alloc_complex(M_loc * N_loc);
		fftw_complex* tempComplex = (fftw_complex*) fftw_alloc_complex(M_loc * N_loc);
		memset(tempSquaring, 0, M_loc * N_loc * sizeof(*tempSquaring));
		memset(tempComplex, 0, M_loc * N_loc * sizeof(*tempComplex));
		/*MPI_Iscatter(*/
				/*a_complex, 1, matSegType,*/
				/*tempSquaring, 1, matSegType,*/
				/*0, commHandle, &requests[0]*/
		/*);*/
		/*MPI_Iscatter(*/
				/*input_complex, 1, matSegType,*/
				/*tempComplex, 1, matSegType,*/
				/*0, commHandle, &requests[1]*/
		/*);*/
		/*MPI_Waitall(2, requests, NULL);*/
		#pragma omp for private(i)
		for (size_t i = 0; i < M_loc * N_loc; i++) {
			tempSquaring[i] *= tempComplex[i];
		}
		/*MPI_Gather(*/
			/*tempSquaring, 1, matSegType,*/
			/*a_complex, 1, matSegType,*/
			/*0, commHandle*/
		/*);*/
		fftw_free(tempSquaring);
		fftw_free(tempComplex);
		mat2 result = calloc(M * N, sizeof(*result));
		fftw_plan plan_result = fftw_plan_dft_c2r_2d(M, N, a_complex, result, FFTW_BACKWARD | FFTW_ESTIMATE);
		fftw_execute(plan_result);
		fftw_destroy_plan(plan_result);
		// TODO: ROTATE RESULT
		#pragma omp for private(i,j) collapse(2)
		for (size_t i = 0; i < M; i++) {
			for (size_t j = 0; j < N; j++) {
				V(u, i, j) = V(result, (i + (timesteps % M)) % M, (j + (timesteps % M)) % M);
			}
		}
	}
	fftw_free(a_complex);
	fftw_free(input_complex);
	cleanupFFTConv();
#else
	fprintf(stderr, "Unsupported, recompile with -DFFT_CONV_KERNEL=1 to use");
#endif
} //ompAdvectExtra()
