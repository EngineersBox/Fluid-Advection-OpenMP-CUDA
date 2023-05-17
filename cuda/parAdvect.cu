// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

#include <cufftw.h>

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //initParParams()


__host__ __device__
static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
  double v2 = v/2.0;
  *cm1 = v2*(v+1.0);
  *c0  = 1.0 - v*v;
  *cp1 = v2*(v-1.0);
}

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

#define EXCHANGE_PARAMS size_t threadId, size_t i, size_t j, double* u, size_t ldu, size_t N_loc, size_t M_loc, size_t M0, size_t N0

void exchangeBlockCorner(EXCHANGE_PARAMS) {
	// Exchange single corner
	size_t horizontalCond = i == 0;
	size_t verticalCond = j == 0;
	#if LOG_2D_EXCHANGES == 1
		printf(
				"[C] [%zu] [%zu,%zu] => [%zu,%zu]\n",
				threadId,
				modAlt(verticalCond, M , 1),
				modAlt(horizontalCond, N, 1),
				modAlt(verticalCond, 0, M + 1),
				modAlt(horizontalCond, 0, N + 1)
		);
	#endif
	V(u, modAlt(verticalCond, 0, M + 1), modAlt(horizontalCond, 0, N + 1)) = V(u, modAlt(verticalCond, M, 1), modAlt(horizontalCond, N, 1));
	// Exchange top or bottom
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	#if LOG_2D_EXCHANGES == 1
		printf("[C] [%zu] yDst: %zu, ySrc: %zu, x: [%zu,%zu)\n", threadId, yDst, ySrc, N0, N0 + N_loc);
	#endif
	for (size_t x = N0; x < N0 + N_loc; x++) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
	// Exchange left or right
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	#if LOG_2D_EXCHANGES == 1
		printf("[C] [%zu] xDst: %zu, xSrc: %zu, y: [%zu,%zu)\n", threadId, xDst, xSrc, M0, M0 + M_loc);
	#endif
	for (size_t y = M0; y < M0 + M_loc; y++) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeBlockTopBottom(EXCHANGE_PARAMS) {
	// Exchange top or bottom
	size_t verticalCond = j == 0;
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	#if LOG_2D_EXCHANGES == 1
		printf("[%s] [%zu] yDst: %zu, ySrc: %zu, x: [%zu,%zu)\n", verticalCond ? "T" : "B", threadId, yDst, ySrc, N0, N0 + N_loc);
	#endif
	for (size_t x = N0; x < N0 + N_loc; x++) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
}

void exchangeBlockLeftRight(EXCHANGE_PARAMS) {
	// Exchange left or right
	size_t horizontalCond = i == 0;
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	#if LOG_2D_EXCHANGES == 1
		printf("[%s] [%zu] xDst: %zu, xSrc: %zu, y: [%zu,%zu)\n", horizontalCond ? "L" : "R", threadId, xDst, xSrc, M0, M0 + M_loc);
	#endif
	for (size_t y = M0; y < M0 + M_loc; y++) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

void exchangeNoop(EXCHANGE_PARAMS) {}

typedef void(*ExchangeHandler)(EXCHANGE_PARAMS);

#define handlerAlt(modi, exchangeHandler) ((ExchangeHandler) modAlt((modi), (uintptr_t) &(exchangeHandler), (uintptr_t) &exchangeNoop))
#define cornerMod(i, j)    (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) == 0)
#define topBottomMod(i, j) (uintptr_t) ((i) % (P - 1) != 0 && (j) % (Q - 1) == 0)
#define leftRightMod(i, j) (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) != 0)

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
	dim3 dimBlock(Bx, By);
	dim3 dimGrid(Gx, Gy);
	for (size_t r = 0; r < reps; r++) {
		Kernel<<<dimGrid, dimBlock>>>(/*params*/);
	}
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

} //cudaOptAdvect()
