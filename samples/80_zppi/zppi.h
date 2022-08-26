#ifndef __ZPPI_H__
#define __ZPPI_H__

#if BUILD_LINUX
#define ZPPI_EXPORT __attribute__ ((visibility ("default")))
#else
#define ZPPI_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

	ZPPI_EXPORT cudaError_t zppiAlphaComp_8u_AP4R_Over(uchar1* pSrc0[4], int src0Step[4], uchar1* pSrc1[4], int src1Step[4], uchar1* pDst[4], int dstStep[4], int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr444_YCbCr420_8u_P3P2R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[2], int dstStep[2], int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr444_YCbCr420_8u_P3R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr420_YCbCrA444_8u_P2P4R(uchar1* pSrc[2], int srcStep[2], uchar1* pDst[4], int dstStep[4], uchar1 nAlpha, int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr420_YCbCrA444_8u_P3P4R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst[4], int dstStep[4], uchar1 nAlpha, int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr422_8u_P1P3R(uchar1* pSrc, int srcStep, uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr422_8u_P3P1R(uchar1* pSrc[3], int srcStep[3], uchar1* pDst, int dstStep, int nWidth, int nHeight);
	ZPPI_EXPORT cudaError_t zppiYCbCr422_YCbCr420_8u_P1P3R(uchar1* pSrc, int srcStep, uchar1* pDst[3], int dstStep[3], int nWidth, int nHeight);

#ifdef __cplusplus
}
#endif

#endif // __ZPPI_H__
