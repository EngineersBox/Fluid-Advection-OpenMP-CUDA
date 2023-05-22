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

#define CHECK_ALLOC(type, var, expr) \
	type var = (type) expr; \
	assert(var != NULL);

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

#if FFT_CONV_KERNEL == 1
static const double CFL = 0.25;

// FFT Optimisation Variables
static double _deltax;
static double _deltay;
static double _dt;

static double Ux;
static double Uy;

// N2 Coefficients
static double cim1;
static double ci0;
static double cip1;
static double cjm1;
static double cj0;
static double cjp1;

static double* laxWendroffKernel;

#define initFFTConv() ({ \
	_deltax = 1.0 / N; \
	_deltay = 1.0 / M; \
	_dt = CFL * (_deltax < _deltay ? _deltax : _deltay); \
	Ux = Velx * _dt / _deltax; \
	Uy = Vely * _dt / _deltay; \
	cim1 = (Ux / 2.0) * (Ux + 1.0); \
	ci0 = 1.0 - (Ux * Ux); \
	cip1 = (Ux / 2.0) * (Ux - 1.0); \
	cjm1 = (Uy / 2.0) * (Uy + 1.0); \
	cj0 = 1.0 - (Uy * Uy); \
	cjp1 = (Uy / 2.0) * (Uy - 1.0); \
	laxWendroffKernel = (double*) calloc(N * M, sizeof(*laxWendroffKernel)); \
	assert(laxWendroffKernel != NULL); \
	memset(laxWendroffKernel, 0, N * N * sizeof(*laxWendroffKernel)); \
	laxWendroffKernel[0] = cim1 * cjm1; \
	laxWendroffKernel[1] = cim1 * cj0; \
	laxWendroffKernel[2] = cim1 * cjp1; \
	laxWendroffKernel[M + 0] = ci0 * cjm1; \
	laxWendroffKernel[M + 1] = ci0 * cj0; \
	laxWendroffKernel[M + 2] = ci0 * cjp1; \
	laxWendroffKernel[(2 * M) + 0] = cip1 * cjm1; \
	laxWendroffKernel[(2 * M) + 1] = cip1 * cj0; \
	laxWendroffKernel[(2 * M) + 2] = cip1 * cjp1; \
	if (!fftw_init_threads()) { \
		fprintf(stderr, "FFTW failed to initialise with OpenMP threads"); \
		exit(1); \
		return; \
	} \
	fftw_plan_with_nthreads(omp_get_max_threads()); \
})

#define cleanupFFTConv() ({ \
	free(laxWendroffKernel); \
	fftw_cleanup_threads(); \
})
#else
#define initFFTConv() ({})
#define cleanupFFTConv() ({})
#endif

static int M, N, P, Q, fieldSize; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
	M = M_, N = N_; P = P_, Q = Q_;
	fieldSize = M * N;
	verbosity = verb;
	initFFTConv();
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

#define modAlt(modi, trueValue, falseValue) (((modi) * (trueValue)) + ((1 - (modi)) * (falseValue)))

#define EXCHANGE_PARAMS size_t threadId, size_t i, size_t j, mat2 u, size_t ldu, size_t N_loc, size_t M_loc, size_t M0, size_t N0
typedef void(*ExchangeHandler)(EXCHANGE_PARAMS);

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
				if ((i) % (P - 1) == 0 && (j) % (Q - 1) == 0) {
					exchangeBlockCorner(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				}
				if ((i) % (P - 1) != 0 && (j) % (Q - 1) == 0) {
					exchangeBlockTopBottom(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				}
				if ((i) % (P - 1) == 0 && (j) % (Q - 1) != 0) {
					exchangeBlockLeftRight(threadId, i, j, u, ldu, N_loc, M_loc, M0, N0);
				}
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
void repeatedSquaring(int timesteps, fftw_complex* a_complex, size_t n) {
	bool initialised = false;
	int t = timesteps;
	CHECK_ALLOC(fftw_complex*, odd_mults, fftw_alloc_complex(n));
	#pragma omp parallel
	{
		while (t > 1) {
			if (t & 1) {
				if (!initialised) {
					memcpy(odd_mults, a_complex, n);
					initialised = true;
				} else {
					#pragma omp for private(i)
					for_r (i, 0, n) {
						odd_mults[i] *= a_complex[i];
					}
				}
			}
			#pragma omp for private(i)
			for_r (i, 0, n) {
				a_complex[i] *= a_complex[i];
			}
			t /= 2;
		}
		if (initialised) {
			#pragma omp for private(i)
			for_r (i, 0, n) {
				a_complex[i] *= odd_mults[i];
			}
		}
	}
	fftw_free(odd_mults);
}
#endif

// ... extra optimization variant
void ompAdvectExtra(int r, mat2 u, int ldu) {
#if FFT_CONV_KERNEL == 1
	size_t timesteps = r;

	// FFT Forward
	CHECK_ALLOC(fftw_complex*, a_complex, fftw_alloc_complex(fieldSize));
	memset(a_complex, 0, sizeof(fftw_complex) * fieldSize);
	fftw_plan plan_u_f = fftw_plan_dft_r2c_2d(M, N, u, a_complex, FFTW_ESTIMATE);
	fftw_execute(plan_u_f);
	fftw_destroy_plan(plan_u_f);
	
	// Repeated squaring
	repeatedSquaring(timesteps, a_complex, fieldSize);

	// FFT Forward
	CHECK_ALLOC(fftw_complex*, input_complex, fftw_alloc_complex(fieldSize));
	fftw_plan plan_lwk_f = fftw_plan_dft_r2c_2d(M, N, laxWendroffKernel, input_complex, FFTW_ESTIMATE);
	fftw_execute(plan_lwk_f);
	fftw_destroy_plan(plan_lwk_f);

	// Combine kernel & rep-squared evolution
	#pragma omp parallel for private(i) shared(a_complex, input_complex)
	for_r (i, 0, fieldSize) {
		a_complex[i] *= input_complex[i];
	}

	// FFT Backward
	CHECK_ALLOC(mat2, result, calloc(fieldSize, sizeof(*result)));
	fftw_plan plan_result = fftw_plan_dft_c2r_2d(M, N, a_complex, result, FFTW_BACKWARD | FFTW_ESTIMATE);
	fftw_execute(plan_result);
	fftw_destroy_plan(plan_result);

	// Rotate result
	#pragma omp parallel for private(i,j) shared(u, result, timesteps, M, N) collapse(2)
	for_r (i, 0, M) {
		for_r (j, 0, N) {
			V(u, i, j) = result[((i + (timesteps % M)) % M) + (((j + (timesteps % N)) % N) * M)];
		}
	}

	fftw_free(a_complex);
	fftw_free(input_complex);
	cleanupFFTConv();
#else
	fprintf(stderr, "Unsupported, recompile with -DFFT_CONV_KERNEL=1 to use");
#endif
} //ompAdvectExtra()
