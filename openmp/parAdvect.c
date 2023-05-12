// OpenMP parallel 2D advection solver module
// template written for COMP4300/8300 Assignment 2, 2021
// template v1.0 14 Apr 

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
	M = M_, N = N_; P = P_, Q = Q_;
	verbosity = verb;
} //initParParams()


__attribute__((always_inline)) static inline void omp1dUpdateBoundary(double *u, int ldu) {
	int i, j;
	#pragma omp for private(j)
	for (j = 1; j < N+1; j++) { //top and bottom halo
		V(u, 0, j)   = V(u, M, j);
		V(u, M+1, j) = V(u, 1, j);
	}
	#pragma omp for private(i)
	for (i = 0; i < M+2; i++) { //left and right sides of halo 
		V(u, i, 0) = V(u, i, N);
		V(u, i, N+1) = V(u, i, 1);
	}
} 


__attribute__((always_inline)) static inline void omp1dUpdateAdvectField(double *u, int ldu, double *v, int ldv) {
	int i, j;
	double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
	double cim1, ci0, cip1, cjm1, cj0, cjp1;
	N2Coeff(Ux, &cim1, &ci0, &cip1); N2Coeff(Uy, &cjm1, &cj0, &cjp1);
	#pragma omp for private(i,j) collapse(2)
	for (i=0; i < M; i++)
		for (j=0; j < N; j++) 
			V(v,i,j) =
				cim1*(cjm1*V(u,i-1,j-1) + cj0*V(u,i-1,j) + cjp1*V(u,i-1,j+1)) +
				ci0 *(cjm1*V(u,i  ,j-1) + cj0*V(u,i,  j) + cjp1*V(u,i,  j+1)) +
				cip1*(cjm1*V(u,i+1,j-1) + cj0*V(u,i+1,j) + cjp1*V(u,i+1,j+1));
} //omp1dUpdateAdvectField()  


__attribute__((always_inline)) static inline void omp1dCopyField(double *v, int ldv, double *u, int ldu) {
	int i, j;
	#pragma omp for private(i,j) collapse(2)
	for (i=0; i < M; i++)
		for (j=0; j < N; j++)
			V(u,i,j) = V(v,i,j);
} //omp1dCopyField()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, double *u, int ldu) {
	int ldv = N+2;
	double *v = calloc(ldv*(M+2), sizeof(double)); assert(v != NULL);
	for (int r = 0; r < reps; r++) {    
		#pragma omp parallel shared(u,ldu,v,ldv)
		{
			omp1dUpdateBoundary(u, ldu);
			omp1dUpdateAdvectField(&V(u,1,1), ldu, &V(v,1,1), ldv);
			omp1dCopyField(&V(v,1,1), ldv, &V(u,1,1), ldu);
		}
	} //for (r...)
	free(v);
} //omp1dAdvect()

/* This avoids the use of if statements which cause warp/wavefront divergence.
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
#define modAlt(modi, trueValue, falseValue) (((modi) * (falseValue)) + ((1 - (modi)) * (trueValue)))

#define EXCHANGE_PARAMS size_t i, size_t j, double* u, size_t ldu, size_t N_loc, size_t M_loc, size_t M0, size_t N0

void exchangeBlockCorner(EXCHANGE_PARAMS) {
	// Exchange single corner
	size_t horizontalCond = i == 0;
	size_t verticalCond = j == 0;
	V(u, modAlt(verticalCond, 0, M + 1), modAlt(horizontalCond, 0, N + 1)) = V(u, modAlt(verticalCond, M, 1), modAlt(horizontalCond, N, 1));
	// Exchange top or bottom
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	for (size_t x = N0; x < N0 + N_loc; x++) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
	// Exchange left or right
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	for (size_t y = M0; y < M0 + M_loc; y++) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeBlockTopBottom(EXCHANGE_PARAMS) {
	// Exchange top or bottom
	size_t verticalCond = j == 0;
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	for (size_t x = N0; x < N0 + N_loc; x++) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
}

void exchangeBlockLeftRight(EXCHANGE_PARAMS) {
	// Exchange left or right
	size_t horizontalCond = i == 0;
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	for (size_t y = M0; y < M0 + M_loc; y++) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeNoop(EXCHANGE_PARAMS) {}

typedef void(*ExchangeHandler)(EXCHANGE_PARAMS);

#define handlerAlt(modi, exchangeHandler) ((ExchangeHandler) modAlt((modi), (uintptr_t) &(exchangeHandler), (uintptr_t) &exchangeNoop))
#define cornerMod(i, j) (uintptr_t) (((i) == 0 || (i) == P) && ((j) == 0 || (j) == Q))
#define topBottomMod(i, j) (uintptr_t) ((i) != 0 && (i) != P && ((j) == 0 || (j) == Q))
#define leftRightMod(i, j) (uintptr_t) (((i) == 0 || (i) == P) && (j) != 0 && (j) != Q)

// ... using 2D parallelization
void omp2dAdvect(int reps, double *u, int ldu) {
	size_t i, j, r, ldv = N+2;
	double *v = calloc(ldv * (M + 2), sizeof(*v));
	assert(v != NULL);

	for (r = 0; r < reps; r++) {
		#pragma omp parallel for shared(u, ldu, v, ldv) private(i, j) collapse(2)
		for (i = 0; i < P; i++) {
			for (j = 0; j < Q; j++) {
				size_t threadId = omp_get_thread_num();
				size_t P0 = threadId / Q;
				size_t M0 = (M / P) * P0;
				size_t M_loc = (P0 < P - 1) ? (M / P) : (M - M0);
				size_t Q0 = threadId % Q;
				size_t N0 = (N / Q) * Q0;
				size_t N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);
				// Pointer arithmetic to avoid conditionals and warp divergence
				handlerAlt(cornerMod(i, j), exchangeBlockCorner)(i, j, u, ldu, N_loc, M_loc, M0, N0);
				handlerAlt(topBottomMod(i, j), exchangeBlockTopBottom)(i, j, u, ldu, N_loc, M_loc, M0, N0);
				handlerAlt(leftRightMod(i, j), exchangeBlockLeftRight)(i, j, u, ldu, N_loc, M_loc, M0, N0);
				updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv);
				copyField(M_loc, N_loc, &V(v, M0, N0), ldv, &V(u, M0, N0), ldu);
			}
		}

/*#pragma omp parallel for shared(u, ldu, v, ldv) private(j)*/
		/*for (j = 1; j < N+1; j++) { //top and bottom halo*/
			/*V(u, 0, j)   = V(u, M, j);*/
			/*V(u, M+1, j) = V(u, 1, j);*/
		/*}*/
/*#pragma omp parallel for shared(u, ldu, v, ldv) private(i)*/
		/*for (i = 0; i < M+2; i++) { //left and right sides of halo */
			/*V(u, i, 0) = V(u, i, N);*/
			/*V(u, i, N+1) = V(u, i, 1);*/
		/*}*/

		/*updateAdvectField(M, N, &V(u,1,1), ldu, &V(v,1,1), ldv);*/

		/*copyField(M, N, &V(v,1,1), ldv, &V(u,1,1), ldu);*/
	} //for (r...)
	free(v);
} //omp2dAdvect()


// ... extra optimization variant
void ompAdvectExtra(int reps, double *u, int ldu) {

} //ompAdvectExtra()
