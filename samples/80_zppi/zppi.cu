#include <stdio.h>
#include "zppi.h"

__global__ void kernel_Copy_8u_C1R(uchar1* pSrc, int srcStep, uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		pDst[y * dstStep + x] = pSrc[y * srcStep + x];
	}
}

ZPPI_EXPORT cudaError_t zppiCopy_8u_C1R(uchar1* pSrc, int srcStep, uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_Copy_8u_C1R<<<grid, block>>>(pSrc, srcStep, pDst, dstStep, nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_AlphaComp_8u_AP4R_Over(uchar1* pSrc00, int src0Step0, uchar1* pSrc01, int src0Step1, uchar1* pSrc02, int src0Step2, uchar1* pSrc03, int src0Step3,
	uchar1* pSrc10, int src1Step0, uchar1* pSrc11, int src1Step1, uchar1* pSrc12, int src1Step2, uchar1* pSrc13, int src1Step3,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, uchar1* pDst3, int dstStep3,
	int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
#if 1
		int a = (int)pSrc03[y * src0Step3 + x].x;
		int a1 = 255 - a;

		pDst0[y * dstStep0 + x].x = ((int)pSrc00[y * src0Step0 + x].x * a + (int)pSrc10[y * src1Step0 + x].x * a1) / 255;
		pDst1[y * dstStep1 + x].x = ((int)pSrc01[y * src0Step1 + x].x * a + (int)pSrc11[y * src1Step1 + x].x * a1) / 255;
		pDst2[y * dstStep2 + x].x = ((int)pSrc02[y * src0Step2 + x].x * a + (int)pSrc12[y * src1Step2 + x].x * a1) / 255;
		pDst3[y * dstStep3 + x].x = (a + (int)pSrc13[y * src1Step3 + x].x * a1) / 255;
#endif

#if 0
		pDst0[y * dstStep0 + x].x = x;
		pDst1[y * dstStep1 + x].x = y;
		pDst2[y * dstStep2 + x].x = x;
		pDst3[y * dstStep3 + x].x = y;
#endif
	}
}

ZPPI_EXPORT cudaError_t zppiAlphaComp_8u_AP4R_Over(uchar1* pSrc0[4], int src0Step[4], uchar1* pSrc1[4], int src1Step[4], uchar1* pDst[4], int dstStep[4], int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

#if 0
	printf("%s(%d): %dx%d, (%p %d %p %d %p %d %p %d) (%p %d %p %d %p %d %p %d) (%p %d %p %d %p %d %p %d)\n",
		__FUNCTION__, __LINE__, nWidth, nHeight,
		pSrc0[0], src0Step[0], pSrc0[1], src0Step[1], pSrc0[2], src0Step[2], pSrc0[3], src0Step[3],
		pSrc1[0], src1Step[0], pSrc1[1], src1Step[1], pSrc1[2], src1Step[2], pSrc1[3], src1Step[3],
		pDst[0], dstStep[0], pDst[1], dstStep[1], pDst[2], dstStep[2], pDst[3], dstStep[3]);
#endif

#if 1
	kernel_AlphaComp_8u_AP4R_Over<<<grid, block>>>(
		pSrc0[0], src0Step[0], pSrc0[1], src0Step[1], pSrc0[2], src0Step[2], pSrc0[3], src0Step[3],
		pSrc1[0], src1Step[0], pSrc1[1], src1Step[1], pSrc1[2], src1Step[2], pSrc1[3], src1Step[3],
		pDst[0], dstStep[0], pDst[1], dstStep[1], pDst[2], dstStep[2], pDst[3], dstStep[3],
		nWidth, nHeight);
#endif

#if 0
	// kernel_Copy_8u_C1R<<<grid, block>>>(pSrc0[0], src0Step[0], pDst[0], dstStep[0], nWidth, nHeight);
	// kernel_Copy_8u_C1R<<<grid, block>>>(pSrc0[1], src0Step[1], pDst[1], dstStep[1], nWidth, nHeight);
	// kernel_Copy_8u_C1R<<<grid, block>>>(pSrc0[2], src0Step[2], pDst[2], dstStep[2], nWidth, nHeight);
	kernel_Copy_8u_C1R<<<grid, block>>>(pSrc0[0], src0Step[0], pDst[1], dstStep[1], nWidth, nHeight);
#endif

	return cudaDeviceSynchronize();
}

__global__ void kernel_YCbCr444_YCbCr420_8u_P3P2R(uchar1* pSrc0, int srcStep0, uchar1* pSrc1, int srcStep1, uchar1* pSrc2, int srcStep2,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int x2 = x * 2;
		int y2 = y * 2;

		// Y
		int nYDstIdx = y2 * dstStep0 + x2;
		int nYSrcIdx = y2 * srcStep0 + x2;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		nYDstIdx += dstStep0;
		nYSrcIdx += srcStep0;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		// CbCr
		int nCbCrDstIdx = y * dstStep1 + x;
		pDst1[nCbCrDstIdx + 0] = pSrc1[y2 * srcStep1 + x2];
		pDst1[nCbCrDstIdx + 1] = pSrc2[y2 * srcStep2 + x2];
	}
}

ZPPI_EXPORT cudaError_t zppiYCbCr444_YCbCr420_8u_P3P2R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[2], int dstStep[2], int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;
	nHeight /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr444_YCbCr420_8u_P3P2R<<<grid, block>>>(
		pSrc[0], srcStep[0],
		pSrc[1], srcStep[1],
		pSrc[2], srcStep[2],
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_YCbCr444_YCbCr420_8u_P3R(uchar1* pSrc0, int srcStep0, uchar1* pSrc1, int srcStep1, uchar1* pSrc2, int srcStep2,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int x2 = x * 2;
		int y2 = y * 2;

		// Y
		int nYDstIdx = y2 * dstStep0 + x2;
		int nYSrcIdx = y2 * srcStep0 + x2;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		nYDstIdx += dstStep0;
		nYSrcIdx += srcStep0;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		// CbCr
		pDst1[y * dstStep1 + x] = pSrc1[y2 * srcStep1 + x2];
		pDst2[y * dstStep2 + x] = pSrc2[y2 * srcStep2 + x2];
	}
}

ZPPI_EXPORT cudaError_t zppiYCbCr444_YCbCr420_8u_P3R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;
	nHeight /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr444_YCbCr420_8u_P3R<<<grid, block>>>(
		pSrc[0], srcStep[0],
		pSrc[1], srcStep[1],
		pSrc[2], srcStep[2],
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		pDst[2], dstStep[2],
		nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_YCbCr420_YCbCrA444_8u_P2P4R(uchar1* pSrc0, int srcStep0, uchar1* pSrc1, int srcStep1,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, uchar1* pDst3, int dstStep3,
	uchar1 nAlpha, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int x2 = x * 2;
		int y2 = y * 2;

		// Y
		int nYDstIdx = y2 * dstStep0 + x2;
		int nYSrcIdx = y2 * srcStep0 + x2;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		nYDstIdx += dstStep0;
		nYSrcIdx += srcStep0;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		// CbCr
		int nCbCrSrcIdx = y * srcStep1 + x;
		uchar1 nCb = pSrc1[nCbCrSrcIdx + 0];
		uchar1 nCr = pSrc1[nCbCrSrcIdx + 1];

		int nCbDstIdx = y2 * dstStep1 + x2;
		int nCrDstIdx = y2 * dstStep2 + x2;
		pDst1[nCbDstIdx + 0] = pDst1[nCbDstIdx + 1] = nCb;
		pDst2[nCrDstIdx + 0] = pDst1[nCrDstIdx + 1] = nCr;

		nCbDstIdx += dstStep1;
		nCrDstIdx += dstStep2;
		pDst1[nCbDstIdx + 0] = pDst1[nCbDstIdx + 1] = nCb;
		pDst2[nCrDstIdx + 0] = pDst1[nCrDstIdx + 1] = nCr;

		// A
		int nADstIdx = y2 * dstStep3 + x2;
		pDst3[nADstIdx + 0] = pDst3[nADstIdx + 1] = nAlpha;

		nADstIdx += dstStep3;
		pDst3[nADstIdx + 0] = pDst3[nADstIdx + 1] = nAlpha;
	}
}

ZPPI_EXPORT cudaError_t zppiYCbCr420_YCbCrA444_8u_P2P4R(uchar1* pSrc[2], int srcStep[2], uchar1* pDst[4], int dstStep[4], uchar1 nAlpha, int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;
	nHeight /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr420_YCbCrA444_8u_P2P4R<<<grid, block>>>(
		pSrc[0], srcStep[0],
		pSrc[1], srcStep[1],
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		pDst[2], dstStep[2],
		pDst[3], dstStep[3],
		nAlpha, nWidth, nHeight);

	return cudaDeviceSynchronize();
}

__global__ void kernel_YCbCr420_YCbCrA444_8u_P3P4R(uchar1* pSrc0, int srcStep0, uchar1* pSrc1, int srcStep1, uchar1* pSrc2, int srcStep2,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, uchar1* pDst3, int dstStep3,
	uchar1 nAlpha, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		int x2 = x * 2;
		int y2 = y * 2;

		// Y
		int nYDstIdx = y2 * dstStep0 + x2;
		int nYSrcIdx = y2 * srcStep0 + x2;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		nYDstIdx += dstStep0;
		nYSrcIdx += srcStep0;
		pDst0[nYDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc0[nYSrcIdx + 1];

		// CbCr
		int nCbCrSrcIdx = y * srcStep1 + x;
		uchar1 nCb = pSrc1[nCbCrSrcIdx];
		uchar1 nCr = pSrc2[nCbCrSrcIdx];

		int nCbDstIdx = y2 * dstStep1 + x2;
		int nCrDstIdx = y2 * dstStep2 + x2;
		pDst1[nCbDstIdx + 0] = pDst1[nCbDstIdx + 1] = nCb;
		pDst2[nCrDstIdx + 0] = pDst1[nCrDstIdx + 1] = nCr;

		nCbDstIdx += dstStep1;
		nCrDstIdx += dstStep2;
		pDst1[nCbDstIdx + 0] = pDst1[nCbDstIdx + 1] = nCb;
		pDst2[nCrDstIdx + 0] = pDst1[nCrDstIdx + 1] = nCr;

		// A
		int nADstIdx = y2 * dstStep3 + x2;
		pDst3[nADstIdx + 0] = pDst3[nADstIdx + 1] = nAlpha;

		nADstIdx += dstStep3;
		pDst3[nADstIdx + 0] = pDst3[nADstIdx + 1] = nAlpha;
	}
}

__global__ void kernel_YCbCr422_8u_P1P3R(uchar1* pSrc, int srcStep,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		// Y
		int nYDstIdx = y * dstStep0 + x * 2;
		int nYCbCrSrcIdx = y * srcStep + x * 4;
		pDst0[nYDstIdx + 0] = pSrc[nYCbCrSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc[nYCbCrSrcIdx + 2];

		// CbCr
		pDst1[y * dstStep1 + x] = pSrc[nYCbCrSrcIdx + 1];
		pDst2[y * dstStep2 + x] = pSrc[nYCbCrSrcIdx + 3];
	}
}

__global__ void kernel_YCbCr422_8u_P3P1R(uchar1* pSrc0, int srcStep0, uchar1* pSrc1, int srcStep1, uchar1* pSrc2, int srcStep2,
	uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		// Y
		int nYCbCrDstIdx = y * dstStep + x * 4;
		int nYSrcIdx = y * srcStep0 + x * 2;
		pDst[nYCbCrDstIdx + 0] = pSrc0[nYSrcIdx + 0];
		pDst[nYCbCrDstIdx + 2] = pSrc0[nYSrcIdx + 1];

		// CbCr
		pDst[nYCbCrDstIdx + 1] = pSrc1[y * srcStep1 + x];
		pDst[nYCbCrDstIdx + 3] = pSrc2[y * srcStep2 + x];
	}
}

__global__ void kernel_YCbCr422_YCbCr420_8u_P1P3R(uchar1* pSrc, int srcStep,
	uchar1* pDst0, int dstStep0, uchar1* pDst1, int dstStep1, uchar1* pDst2, int dstStep2, int nWidth, int nHeight) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if(x < nWidth && y < nHeight) {
		// Y
		int nYDstIdx = y * dstStep0 + x * 2;
		int nYCbCrSrcIdx = y * srcStep + x * 4;
		pDst0[nYDstIdx + 0] = pSrc[nYCbCrSrcIdx + 0];
		pDst0[nYDstIdx + 1] = pSrc[nYCbCrSrcIdx + 2];

		// CbCr
		if(y % 2 == 0) {
			int y2 = y / 2;
			pDst1[y2 * dstStep1 + x] = pSrc[nYCbCrSrcIdx + 1];
			pDst2[y2 * dstStep2 + x] = pSrc[nYCbCrSrcIdx + 3];
		}
	}
}

ZPPI_EXPORT cudaError_t zppiYCbCr420_YCbCrA444_8u_P3P4R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[4], int dstStep[4], uchar1 nAlpha, int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;
	nHeight /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr420_YCbCrA444_8u_P3P4R<<<grid, block>>>(
		pSrc[0], srcStep[0],
		pSrc[1], srcStep[1],
		pSrc[2], srcStep[2],
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		pDst[2], dstStep[2],
		pDst[3], dstStep[3],
		nAlpha, nWidth, nHeight);

	return cudaDeviceSynchronize();
}

ZPPI_EXPORT cudaError_t zppiYCbCr422_8u_P1P3R(uchar1* pSrc, int srcStep, uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr422_8u_P1P3R<<<grid, block>>>(
		pSrc, srcStep,
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		pDst[2], dstStep[2],
		nWidth, nHeight);

	return cudaDeviceSynchronize();
}

ZPPI_EXPORT cudaError_t zppiYCbCr422_8u_P3P1R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst, int dstStep, int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr422_8u_P3P1R<<<grid, block>>>(
		pSrc[0], srcStep[0],
		pSrc[1], srcStep[1],
		pSrc[2], srcStep[2],
		pDst, dstStep,
		nWidth, nHeight);

	return cudaDeviceSynchronize();
}

ZPPI_EXPORT cudaError_t zppiYCbCr422_YCbCr420_8u_P1P3R(uchar1* pSrc, int srcStep, uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight) {
	static int BLOCK_W = 16;
	static int BLOCK_H = 16;

	nWidth /= 2;

	dim3 grid((nWidth + BLOCK_W-1) / BLOCK_W, (nHeight + BLOCK_H-1) / BLOCK_H, 1);
	dim3 block(BLOCK_W, BLOCK_H, 1);

	kernel_YCbCr422_YCbCr420_8u_P1P3R<<<grid, block>>>(
		pSrc, srcStep,
		pDst[0], dstStep[0],
		pDst[1], dstStep[1],
		pDst[2], dstStep[2],
		nWidth, nHeight);

	return cudaDeviceSynchronize();
}
