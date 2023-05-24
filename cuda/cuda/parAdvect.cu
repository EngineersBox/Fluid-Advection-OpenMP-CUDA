// CUDA parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2021
// v1.0 15 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "serAdvect.h" // advection parameters

// ==== CONSTRUCT DEFINES ==== 

#define STATIC_INLINE __attribute__((always_inline)) static inline
#define mat2 double*
#define mat2_r mat2 __restrict__

#define for_rt(T, var, lower, upper) for (T var = (lower); (var) < (upper); (var)++)
#define for_rtc(T, var, lower, upper) for_rt(T, var, lower, (lower) + (upper))
#define for_r(var, lower, upper) for_rt(int, var, lower, upper)
#define for_rc(var, lower, upper) for_rt(int, var, lower, (lower) + (upper))
#define for_ru(var, lower, upper) for_rt(, var, lower, upper)
#define for_rcu(var, lower, upper) for_rt(, var, lower, (lower) + (upper))

// ==== BEHAVIOURAL DEFINES ====

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
static double Ux, Uy;
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

void calcBoundaryWorkSize(int totalEdgeElements) {
	boundaryWorkSizeX = max(1, N / (boundaryGx * boundaryBx));
	boundaryWorkSizeY = max(1, M / (boundaryGy * boundaryBy));
}

void calcBoundaryParams() {
	int totalEdgeElements = (M * 2) + ((N - 2) * 2);
	int totalThreads = Gx * Gy * Bx * By; // Is this right?
	if (totalEdgeElements >= totalThreads) {
		boundaryGx = Gx;
		boundaryGy = Gy;
		boundaryBx = Bx;
		boundaryBy = By;
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	int maxThreads = totalEdgeElements;
	if (maxThreads < Bx * By) {
		boundaryGx = boundaryGy = 1;
		boundaryBx = maxThreads % Bx;
		boundaryBy = max(maxThreads / Bx, By);
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	boundaryBx = Bx;
	boundaryBy = By;
	int blocks = (int) ceilf((float) maxThreads / (float) (Bx * By));
	if (blocks < Gx * Gy) {
		boundaryGx = blocks % Gx;
		boundaryGy = max(blocks / Gx, Gy);
		calcBoundaryWorkSize(totalEdgeElements);
		return;
	}
	boundaryGx = Gx;
	boundaryGy = Gy;

}

#define _h2d(var) HANDLE_ERROR(cudaMemcpyToSymbol(&(dev_##var), &(var), sizeof(var)))

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, int verb) {
	M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
	verbosity = verb;
	calcBoundaryParams();
	printf(
		"Boundary params: Grid=(%d,%d) Block=(%d,%d)\n",
		boundaryGx, boundaryGy,
		boundaryBx, boundaryBy
	);
	Ux = Velx * dt / deltax;
	Uy = Vely * dt / deltay;
} //initParParams()

#define THREAD_LOCALS_LAYOUT(_M, _N, off) \
	int blockId = blockIdx.x + blockIdx.y * gridDim.x; \
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x; \
	int P = gridDim.x * blockDim.x; \
	int Q = gridDim.y * blockDim.y; \
	int x = threadId % Q; \
	int M0 = (((_M) / P) * x) + (off); \
	int M_loc = (x < P - 1) ? ((_M) / P) : ((_M) - M0); \
	int y = threadId / Q; \
	int N0 = (((_N) / Q) * y) + (off); \
	int N_loc = (y < Q - 1) ? ((_N) / Q) : ((_N) - N0)

__host__ __device__ static void N2Coeff(double v, double *cm1, double *c0, double *cp1) {
	double v2 = v/2.0;
	*cm1 = v2*(v+1.0);
	*c0  = 1.0 - v*v;
	*cp1 = v2*(v-1.0);
}

#define DEV_DATA int dev_N, int dev_M, int dev_Ux, int dev_Uy
#define HOST_DATA N, M, Ux, Uy
#define FWD_DEV_DATA dev_N, dev_M, dev_Ux, dev_Uy

__global__ void updateBoundaryNSBlock(double *u, int ldu, DEV_DATA) {
	THREAD_LOCALS_LAYOUT(dev_M, dev_N, 1);
	for_r (j, N0, N0 + N_loc) {
		V(u, 0, j) = V(u, dev_M, j);
		V(u, dev_M + 1, j) = V(u, 1, j);
	}
}

__global__ void updateBoundaryEWBlock(double *u, int ldu, DEV_DATA) {
	THREAD_LOCALS_LAYOUT(dev_M + 2, dev_N, 0);
	for_r (i, M0, M0 + M_loc) {
		V(u, i, 0) = V(u, i, dev_N);
		V(u, i, dev_N + 1) = V(u, i, 1);
	}
}

__global__ void updateAdvectFieldBlock(double *u, int ldu, double *v, int ldv, DEV_DATA) {
	THREAD_LOCALS_LAYOUT(dev_M, dev_N, 1);
	updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv, dev_Ux, dev_Uy);
}

__global__ void copyFieldBlock(double *u, int ldu, double *v, int ldv, DEV_DATA) {
	THREAD_LOCALS_LAYOUT(dev_M, dev_N, 1);
	copyField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv);
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
	double Ux = Velx * dt / deltax, Uy = Vely * dt / deltay;
	int ldv = N+2; double *v;
	HANDLE_ERROR(cudaMalloc(&v, ldv*(M+2)*sizeof(double)));
	mat2 dev_u;
	HANDLE_ERROR(cudaMalloc(&dev_u, ldv * (M + 2) * sizeof(double)));
	HANDLE_ERROR(cudaMemcpy(dev_u, u, ldv * (M + 2) * sizeof(double), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	dim3 dimBlock(Bx,By);
	dim3 dimGrid(Gx,Gy);
	for (int r = 0; r < reps; r++) {
		updateBoundaryNSBlock<<<(Gx,1),(Bx,1)>>>(dev_u, ldu, HOST_DATA);
		updateBoundaryEWBlock<<<(1,Gy),(1,By)>>>(dev_u, ldu, HOST_DATA);
		updateAdvectFieldBlock<<<dimGrid,dimBlock>>>(dev_u, ldu, v, ldv, HOST_DATA);
		copyFieldBlock<<<dimGrid,dimBlock>>>(v, ldv, dev_u, ldu, HOST_DATA);
	} //for(r...)
	HANDLE_ERROR(cudaFree(dev_u));
	HANDLE_ERROR(cudaFree(v));
} //cuda2DAdvect()

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

#define EXCHANGE_PARAMS int i, int j, mat2 u, int ldu, int N_loc, int M_loc, int M0, int N0, DEV_DATA
typedef void (*ExchangeHandler)(EXCHANGE_PARAMS);

__device__ void exchangeBlockTopBottom(EXCHANGE_PARAMS) {
	// Exchange top or bottom
	int verticalCond = i == 0;
	int yDst = modAlt(verticalCond, 0, dev_M + 1);
	int ySrc = modAlt(verticalCond, dev_M, 1);
	for_rc (x, N0, N_loc) {
		V(u, yDst, x) = V(u, ySrc, x);
	}
}

__device__ void exchangeBlockLeftRight(EXCHANGE_PARAMS) {
	// Exchange left or right
	int horizontalCond = j == 0;
	int xDst = modAlt(horizontalCond, 0, dev_N + 1);
	int xSrc = modAlt(horizontalCond, dev_N, 1);
	for_rc (y, M0, M_loc) {
		V(u, y, xDst) = V(u, y, xSrc);
	}
}

__device__ void exchangeBlockCorner(EXCHANGE_PARAMS) {
	// Exchange single corner
	int horizontalCond = j == 0;
	int verticalCond = i == 0;
	V(u, modAlt(verticalCond, 0, dev_M + 1), modAlt(horizontalCond, 0, dev_N + 1)) = V(u, modAlt(verticalCond, dev_M, 1), modAlt(horizontalCond, dev_N, 1));
	// Exchange top or bottom
	exchangeBlockTopBottom(i, j, u, ldu, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	// Exchange left or right
	exchangeBlockLeftRight(i, j, u, ldu, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
}

__device__ void exchangeNoop(EXCHANGE_PARAMS) {}

#define exchangeHandlerAlt(modi, exchangeHandler) ((ExchangeHandler) modAlt((modi), (uintptr_t) &(exchangeHandler), (uintptr_t) &exchangeNoop))
#define cornerMod(i, j)    (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) == 0)
#define topBottomMod(i, j) (uintptr_t) ((i) % (P - 1) != 0 && (j) % (Q - 1) == 0)
#define leftRightMod(i, j) (uintptr_t) ((i) % (P - 1) == 0 && (j) % (Q - 1) != 0)

#define UPDATE_PARAMS int i, int j, mat2 u, int ldu, mat2 v, int ldv, int N_loc, int M_loc, int M0, int N0, DEV_DATA
typedef void (*UpdateHandler)(UPDATE_PARAMS);

__device__ void updateCorner(UPDATE_PARAMS) {
	// Update single corner
	int horizontalCond = i == 0;
	int verticalCond = j == 0;
	#define cornerRef(arr) &V(arr, modAlt(verticalCond, 1, dev_M), modAlt(horizontalCond, 1, dev_N))
	updateAdvectField(1, 1, cornerRef(u), ldu, cornerRef(v), ldv, dev_Ux, dev_Uy);
	// Update top or bottom
	if (N_loc > 1) {
		#define topBottomRef(arr) &V(arr, modAlt(verticalCond, M0, dev_M), modAlt(horizontalCond, N0 + 1, N0))
		updateAdvectField(1, N_loc - 1, topBottomRef(u), ldu, topBottomRef(v), ldv, dev_Ux, dev_Uy);
		#undef topBottomRef
	}
	// Update left or right
	if (M_loc > 1) {
		#define leftRightRef(arr) &V(arr, modAlt(horizontalCond, M0 + 1, M0), modAlt(verticalCond, N0, dev_N))
		updateAdvectField(M_loc - 1, 1, leftRightRef(u), ldu, leftRightRef(v), ldv, dev_Ux, dev_Uy);
		#undef leftRightRef
	}
}

__device__ void updateTopBottom(UPDATE_PARAMS) {
	int verticalCond = j == 0;
	#define topBottomRef(arr) &V(arr, modAlt(verticalCond, M0, dev_M), N0)
	updateAdvectField(1, N_loc, topBottomRef(u), ldu, topBottomRef(v), ldv, dev_Ux, dev_Uy);
}

__device__ void updateLeftRight(UPDATE_PARAMS) {
	int horizontalCond = i == 0;
	#define leftRightRef(arr) &V(arr, M0, modAlt(horizontalCond, N0, dev_N))
	updateAdvectField(M_loc, 1, leftRightRef(u), ldu, leftRightRef(v), ldv, dev_Ux, dev_Uy);
}

__device__ void updateNoop(UPDATE_PARAMS) {}

#define updateHandlerAlt(modi, updateHandler) ((UpdateHandler) modAlt((modi), (uintptr_t) &(updateHandler), (uintptr_t) &updateNoop))

#define KERNEL_PARAMS mat2_r u, int ldu, mat2_r v, int ldv, DEV_DATA

__global__ void updateBoundaries(KERNEL_PARAMS) {
	THREAD_LOCALS_LAYOUT(dev_M, dev_N, 1);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[%d] y: %d, x: %d, P: %d, Q: %d, M: %d, N: %d, M0: %d, N0: %d, M_loc: %d, N_loc: %d\n",
			threadId,
			y, x,
			P, Q,
			dev_M, dev_N,
			M0, N0,
			M_loc, N_loc
		);
	);
	// Pointer arithmetic to avoid conditionals and warp divergence on invocations, reducing the overall diveregence time
	ExchangeHandler exCorner = exchangeHandlerAlt(cornerMod(x, y), exchangeBlockCorner);
	exCorner(x, y, u, ldu, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	ExchangeHandler exTopBottom = exchangeHandlerAlt(topBottomMod(x, y), exchangeBlockTopBottom);
	exTopBottom(x, y, u, ldu, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	ExchangeHandler exLeftRight = exchangeHandlerAlt(leftRightMod(x, y), exchangeBlockLeftRight);
	exLeftRight(x, y, u, ldu, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[%d] i = %d, j = %d, corner: %p, topBottom: %p, leftRight: %p\n",
			threadId,
			y, x,
			(void*) ((uintptr_t) (exCorner != exchangeNoop) * (uintptr_t) exCorner),
			(void*) ((uintptr_t) (exTopBottom != exchangeNoop) * (uintptr_t) exTopBottom),
			(void*) ((uintptr_t) (exLeftRight != exchangeNoop) * (uintptr_t) exLeftRight)
		);
	);
	UpdateHandler upCorner = updateHandlerAlt(cornerMod(x, y), updateCorner);
	upCorner(x, y, u, ldu, v, ldv, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	UpdateHandler upTopBottom = updateHandlerAlt(topBottomMod(x, y), updateTopBottom);
	upTopBottom(x, y, u, ldu, v, ldv, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	UpdateHandler upLeftRight = updateHandlerAlt(leftRightMod(x, y), updateLeftRight);
	upLeftRight(x, y, u, ldu, v, ldv, N_loc, M_loc, M0, N0, FWD_DEV_DATA);
	COMPILE_COND_T(EXCHANGE,
		printf(
			"[%d] i = %d, j = %d, corner: %p, topBottom: %p, leftRight: %p\n",
			threadId,
			y, x,
			(void*) ((uintptr_t) (upCorner != updateNoop) * (uintptr_t) upCorner),
			(void*) ((uintptr_t) (upTopBottom != updateNoop) * (uintptr_t) upTopBottom),
			(void*) ((uintptr_t) (upLeftRight != updateNoop) * (uintptr_t) upLeftRight)
		);
	);
}

__global__ void updateInnerAdvectFieldDevice(KERNEL_PARAMS) {
	THREAD_LOCALS_LAYOUT(dev_M - 1, dev_N - 1, 2);
	int globalIdx = x + y * blockDim.x * gridDim.x;
	updateAdvectField(M_loc, N_loc, &V(u, M0, N0), ldu, &V(v, M0, N0), ldv, dev_Ux, dev_Uy);
}

// ... optimized parallel variant
void cudaOptAdvect(int reps, double *u, int ldu, int w) {
	int ldv = N + 2;
	mat2 v;
	HANDLE_ERROR(cudaMalloc(&v, (ldv * (M + 2)) * sizeof(*v)));
	mat2 uDev;
	HANDLE_ERROR(cudaMalloc(&uDev, (ldv * (M + 2)) * sizeof(*uDev)));
	HANDLE_ERROR(cudaMemcpy(uDev, u, ldv * (M + 2) * sizeof(*u), cudaMemcpyHostToDevice));
	dim3 dimGrid(Gx, Gy);
	dim3 dimBlock(Bx, By);
	// Allow for concurrent exection of boundary updates alongside inner field updates.
	// We need to ensure limited resource usage otherwise concurrency will not be seen
	// in the interleaving of both streams.
	cudaStream_t boundaryUpdateStream;
	cudaStream_t innerFieldAdvectionStream;
	HANDLE_ERROR(cudaStreamCreate(&boundaryUpdateStream));
	HANDLE_ERROR(cudaStreamCreate(&innerFieldAdvectionStream));
	dim3 boundaryGrid(boundaryGx, boundaryGy);
	dim3 boundaryBlock(boundaryBx, boundaryBy);
	cudaDeviceSynchronize();
	for (int r = 0; r < reps; r++) {
		updateBoundaries<<<boundaryGrid, boundaryBlock, 0, boundaryUpdateStream>>>(uDev, ldu, v, ldv, HOST_DATA);
		updateInnerAdvectFieldDevice<<<dimGrid, dimBlock, 0, innerFieldAdvectionStream>>>(uDev, ldu, v, ldv, HOST_DATA);
		cudaDeviceSynchronize(); // Wait for both streams to complete
		COMPILE_COND_T(SWAP, swap(uDev, v, mat2)); // No need for copies, device pointer swaps are sufficient
	}
	COMPILE_COND_T(SWAP,
		if (reps % 2 != 0) {
			swap(uDev, v, mat2);
		}
	);
	HANDLE_ERROR(cudaStreamDestroy(boundaryUpdateStream));
	HANDLE_ERROR(cudaStreamDestroy(innerFieldAdvectionStream));
	HANDLE_ERROR(cudaMemcpy(u, uDev, ldv * (M + 2) * sizeof(*u), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(uDev));
	HANDLE_ERROR(cudaFree(v));
} //cuda2DAdvect()
