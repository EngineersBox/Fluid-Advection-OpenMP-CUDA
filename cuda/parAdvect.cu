// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "serAdvect.h" // advection parameters

#include <cufftw.h>

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

#ifndef LOG_2D_EXCHANGES
#define LOG_2D_EXCHANGES 1
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

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;
static int boundaryGx;
static int boundaryGy;
static int boundaryBx;
static int boundaryBy;
static int boundaryWorkSizeX;
static int boundaryWorkSizeY;

#define _compare(a,b,op) ({ \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a op _b ? _a : _b; \
})

#define max(a,b) _compare(a,b,>)
#define min(a,b) _compare(a,b,<)

void calcBoundaryWorkSize(size_t totalEdgeElements) {
	boundaryWorkSizeX = max(1, N / (boundaryGx * boundaryBx));
	boundaryWorkSizeY = max(1, M / (boundaryGy * boundaryBy));
}

void calcBoundaryParams() {
	size_t totalEdgeElements = (M * 2) + ((N - 2) * 2);
	size_t totalThreads = Gx * Gy * Bx * By; // Is this right?
	if (totalEdgeElements >= totalThreads) {
		boundaryGx = Gx;
		boundaryGy = Gy;
		boundaryBx = Bx;
		boundaryBy = By;
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	size_t maxThreads = totalEdgeElements;
	if (maxThreads < Bx * By) {
		boundaryGx = boundaryGy = 1;
		boundaryBx = maxThreads % Bx;
		boundaryBy = max(maxThreads / Bx, By);
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	boundaryBx = Bx;
	boundaryBy = By;
	size_t blocks = (size_t) ceilf((float) maxThreads / (float) (Bx * By));
	if (blocks < Gx * Gy) {
		boundaryGx = blocks % Gx;
		boundaryGy = max(blocks / Gx, Gy);
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	boundaryGx = Gx;
	boundaryGy = Gy;

}

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, int verb) {
	M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
	verbosity = verb;
	calcBoundaryParams();
} //initParParams()


__host__ __device__ static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
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

#define EXCHANGE_PARAMS size_t threadId, size_t i, size_t j, mat2 u, size_t ldu, size_t N_loc, size_t M_loc, size_t M0, size_t N0

__device__ void exchangeBlockCorner(EXCHANGE_PARAMS) {
	// Exchange single corner
	size_t horizontalCond = i == 0;
	size_t verticalCond = j == 0;
	V(u, modAlt(verticalCond, 0, M + 1), modAlt(horizontalCond, 0, N + 1)) = V(u, modAlt(verticalCond, M, 1), modAlt(horizontalCond, N, 1));
	// Exchange top or bottom
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	for_rc (x, N0, N_loc) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
	// Exchange left or right
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	for_rc (y, M0, M_loc) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

__device__ void exchangeBlockTopBottom(EXCHANGE_PARAMS) {
	// Exchange top or bottom
	size_t verticalCond = j == 0;
	size_t yDst = modAlt(verticalCond, 0, M + 1);
	size_t ySrc = modAlt(verticalCond, M, 1);
	for_rc (x, N0, N_loc) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
}

__device__ void exchangeBlockLeftRight(EXCHANGE_PARAMS) {
	// Exchange left or right
	size_t horizontalCond = i == 0;
	size_t xDst = modAlt(horizontalCond, 0, N + 1);
	size_t xSrc = modAlt(horizontalCond, N, 1);
	for_rc (y, M0, M_loc) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

__device__ void exchangeNoop(EXCHANGE_PARAMS) {}

typedef void(*ExchangeHandler)(EXCHANGE_PARAMS);

#define handlerAlt(modi, exchangeHandler) ((ExchangeHandler) modAlt((modi), (uintptr_t) &(exchangeHandler), (uintptr_t) &exchangeNoop))
#define cornerMod(i, j)    (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) == 0)
#define topBottomMod(i, j) (uintptr_t) ((i) % (P - 1) != 0 && (j) % (Q - 1) == 0)
#define leftRightMod(i, j) (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) != 0)

#define THREAD_LOCALS_LAYOUT \
	size_t x = threadIdx.x + blockIdx.x * blockDim.x; \
	size_t y = threadIdx.y + blockIdx.y * blockDim.y; \
	size_t offset = x + y * blockDim.x * gridDim.x; \
	size_t P0 = threadId / Q; \
	size_t M0 = ((M / P) * P0) + 1; \
	size_t M_loc = (P0 < P - 1) ? (M / P) : (M - M0); \
	size_t Q0 = threadId % Q; \
	size_t N0 = ((N / Q) * Q0) + 1; \
	size_t N_loc = (Q0 < Q - 1) ? (N / Q) : (N - N0);

#define KERNEL_PARAMS __shared__ mat2_r u, size_t ldu, __shared__ mat2_r v, size_t ldv

__global__ void updateBoundaries(KERNEL_PARAMS) {
	THREAD_LOCALS_LAYOUT
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
	)
	updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv);
	// Update top/bottom (full) if at top or bottom
	// Update left/right (without top or bottom, if it at corner) if at left or right
}

__global__ void updateInnerAdvectFieldDevice(KERNEL_PARAMS) {
	//THREAD_LOCALS_LAYOUT
	size_t x = threadIdx.x + blockIdx.x * blockDim.x;
	size_t y = threadIdx.y + blockIdx.y * blockDim.y;
	size_t globalIdx = x + y * blockDim.x * gridDim.x;
	updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
	size_t ldv = N + 2;
	mat2 v = cudaMalloc(ldv * (M + 2), sizeof(*v));
	assert(v != NULL);
	mat2 uDev = cudaMalloc(ldv * (M + 2), sizeof(*uDev));
	assert(uDev != NULL);
	cudaMemcpy(uDev, u, ldv * (M + 2) * sizeof(*u), cudaMemcpyHostToDevice);
	dim3 dimGrid(Gx, Gy);
	dim3 dimBlock(Bx, By);
	// Allow for concurrent exection of boundary updates alongside inner field updates.
	// We need to ensure limited resource usage otherwise concurrency will not be seen
	// in the interleaving of both streams.
	cudaStream_t boundaryUpdateStream;
	cudaStream_t innerFieldAdvectionStream;
	dim3 boundaryGrid(boundaryGx, boundaryGy);
	dim3 boundaryBlock(boundaryBx, boundaryBy);
	for (size_t r = 0; r < reps; r++) {
		updateBoundaries<<<boundaryGrid, boundaryBlock, boundaryUpdateStream>>>(uDev, ldu, v, ldv);
		updateInnerAdvectFieldDevice<<<dimGrid, dimBlock, innerFieldAdvectionStream>>>(uDev, ldu, v, ldv);
		cudaDeviceSynchronize(); // Wait for both streams to complete
		COMPILE_COND_T(SWAP, swap(uDev, v, mat2)); // No need for copies, device pointer swaps are sufficient
	}
	cudaMemcpy(u, uDev, ldv * (M + 2) * sizeof(*u), cudaMemcpyDeviceToHost);
	cudaFree(uDev);
	cudaFree(v);
} //cuda2DAdvect()



// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {

} //cudaOptAdvect()
