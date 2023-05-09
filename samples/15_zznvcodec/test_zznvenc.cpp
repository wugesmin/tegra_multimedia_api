#include "zznvcodec.h"
#include "ZzLog.h"
#include <unistd.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <npp.h>
#include <cuda_runtime.h>

#define OutputFile

ZZ_INIT_LOG("test_zznvenc");

uint8_t* zziMalloc_8u_C1(int width, int height, int* step) {
	*step = width;
	return (uint8_t*)aligned_alloc(16, width * height);
}

void zziFree(void* p) {
	if(p) free(p);
}

std::ofstream of_bits;

void _zznvcodec_encoder_on_video_packet(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp, intptr_t pUser) {
	LOGD("pBuffer=%p, nSize=%d, nFlags=%d, nTimestamp=%.2f", pBuffer, nSize, nFlags, nTimestamp / 1000.0);
#ifdef OutputFile
	of_bits.write((const char*)pBuffer, nSize);
#endif	
}

int main(int argc, char *argv[])
{
#ifdef 	OutputFile
	FILE *fp;
	FILE *pOutputTxtFile;
	fp = fopen("test_av1_2_cb.yuv","rb");
	pOutputTxtFile = fopen("AV14KTxt", "w");
	char cDataSize[256];
#endif	
	of_bits.open("AV14KTxt.av1", std::ios::binary);

	int nWidth = 3840;
	int nHeight = 2160;
	unsigned char *pOutBuffer = NULL;

	zznvcodec_encoder_t* pEnc = zznvcodec_encoder_new();

#if 0
	zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_NV24;
#else
	zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_NV12;
#endif

	zznvcodec_encoder_set_video_property(pEnc, nWidth, nHeight, nPixFmt);
	zznvcodec_pixel_format_t nEncoderPixFmt = ZZNVCODEC_CODEC_TYPE_AV1;
	zznvcodec_encoder_set_misc_property(pEnc, ZZNVCODEC_PROP_ENCODER_PIX_FMT, (intptr_t)&nEncoderPixFmt);
#ifndef DIRECT_OUTPUT	
	zznvcodec_encoder_register_callbacks(pEnc, _zznvcodec_encoder_on_video_packet, (intptr_t)0);
#endif	
	zznvcodec_encoder_start(pEnc);

	zznvcodec_video_frame_t oVideoFrame;
	memset(&oVideoFrame, 0, sizeof(oVideoFrame));

	zznvcodec_video_plane_t& plane0 = oVideoFrame.planes[0];
	zznvcodec_video_plane_t& plane1 = oVideoFrame.planes[1];
	zznvcodec_video_plane_t& plane2 = oVideoFrame.planes[2];

	switch(nPixFmt) {
		
	case ZZNVCODEC_PIXEL_FORMAT_NV12:
		oVideoFrame.num_planes = 2;
		plane0.width = nWidth;
		plane0.height = nHeight;

		plane0.ptr = zziMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);

		LOGI("plane0.ptr = %p / %d", plane0.ptr, plane0.stride);

		plane1.width = nWidth;
		plane1.height = nHeight/2;

		plane1.ptr = zziMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);

		LOGI("plane1.ptr = %p / %d", plane1.ptr, plane1.stride);
		break;			
		
	case ZZNVCODEC_PIXEL_FORMAT_NV24:
		oVideoFrame.num_planes = 2;
		plane0.width = nWidth;
		plane0.height = nHeight;

		plane0.ptr = zziMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);

		LOGI("plane0.ptr = %p / %d", plane0.ptr, plane0.stride);

		plane1.width = nWidth*2;
		plane1.height = nHeight;

		plane1.ptr = zziMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);

		LOGI("plane1.ptr = %p / %d", plane1.ptr, plane1.stride);
		break;		

	case ZZNVCODEC_PIXEL_FORMAT_YUV420P:
		oVideoFrame.num_planes = 3;
		plane0.width = nWidth;
		plane0.height = nHeight;
	#if 0 // USE_CUDA_MEMORY
		plane0.ptr = nppiMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);
	#else
		plane0.ptr = zziMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);
	#endif
		LOGI("plane0.ptr = %p / %d", plane0.ptr, plane0.stride);

		plane1.width = nWidth / 2;
		plane1.height = nHeight / 2;
	#if 0 // USE_CUDA_MEMORY
		plane1.ptr = nppiMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);
	#else
		plane1.ptr = zziMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);
	#endif
		LOGI("plane1.ptr = %p / %d", plane1.ptr, plane1.stride);

		plane2.width = nWidth / 2;
		plane2.height = nHeight / 2;
	#if 0 // USE_CUDA_MEMORY
		plane2.ptr = nppiMalloc_8u_C1(plane2.width, plane2.height, &plane2.stride);
	#else
		plane2.ptr = zziMalloc_8u_C1(plane2.width, plane2.height, &plane2.stride);
	#endif
		LOGI("plane2.ptr = %p / %d", plane2.ptr, plane2.stride);
		break;

	default:
		LOGE("%s(%d): unexpected value, nPixFmt=%X", __FUNCTION__, __LINE__, nPixFmt);
		break;
	}

	int nFPS = 60;

#if (defined OutputFile) && (defined DIRECT_OUTPUT)		
	pOutBuffer = (unsigned char*) malloc(nWidth*nHeight*3 * sizeof(unsigned char));			
#endif	

	for(int i = 0;i < 100;++i) {
#ifdef OutputFile
		for (int i =0 ; i< plane0.height ; i++) {
			fread( plane0.ptr + i * plane0.stride, 1, plane0.width, fp);
		}
		//LOGI("plane1.w = %d / plane1.stride %d", plane1.width, plane1.stride);	
		for (int i =0 ; i< plane1.height ; i++) {
			fread( plane1.ptr + i * plane1.stride, 1, plane1.width, fp);
		}	
		
		if ( oVideoFrame.num_planes == 3) {
			for (int i =0 ; i< plane2.height ; i++) {
				fread( plane2.ptr + i * plane2.stride, 1, plane2.width, fp);
			}	
		} 
#endif

		LOGI("Frame %d", i);
		int nOutSize = 0;
		int64_t nOutTimeStamp = 0;

		zznvcodec_encoder_set_video_uncompression_buffer(pEnc, &oVideoFrame, i * 1000000L / nFPS, pOutBuffer, &nOutSize, &nOutTimeStamp);

#if (defined OutputFile) && (defined DIRECT_OUTPUT)		
		// Direct Output
		if (nOutSize != 0) {
			LOGD("%s(%d): ,outsize: %d timestamp: %.2f  Outbuffer:%p OutFPTxt:%p\n", __FUNCTION__, __LINE__, nOutSize, nOutTimeStamp / 1000.0,pOutBuffer, pOutputTxtFile);	
			of_bits.write((const char*)pOutBuffer, nOutSize);
			sprintf(cDataSize,"%d", nOutSize);
			fputs(cDataSize, pOutputTxtFile);
			fwrite("\r\n", 1, 2, pOutputTxtFile);
		}			

#endif		
	}

#if (defined OutputFile) && (defined DIRECT_OUTPUT)	
	// Flush Frame
	while (1)
	{
		LOGD("%s(%d): , Flush frame start\n", __FUNCTION__, __LINE__);	
		int nOutSize = 0;
		int64_t nOutTimeStamp = 0;		
		zznvcodec_encoder_set_video_uncompression_buffer(pEnc, NULL, 0, pOutBuffer, &nOutSize, &nOutTimeStamp);	
		// Direct Output	
		if (nOutSize != 0) {
			LOGD("%s(%d): , flush frame %.2f\n", __FUNCTION__, __LINE__, nOutTimeStamp / 1000.0);	
			of_bits.write((const char*)pOutBuffer, nOutSize);
		}
		else
			break;
	}
	free(pOutBuffer);		
	pOutBuffer = NULL;

#endif
	zznvcodec_encoder_stop(pEnc);
	zznvcodec_encoder_delete(pEnc);
	pEnc = NULL;

#if 1 // USE_CUDA_MEMORY
	if(plane0.ptr) {
		nppiFree(plane0.ptr);
	}
	if(plane1.ptr) {
		nppiFree(plane1.ptr);
	}
	if(plane2.ptr) {
		nppiFree(plane2.ptr);
	}
#else
	zziFree(plane0.ptr);
	zziFree(plane1.ptr);
	zziFree(plane2.ptr);
#endif
	plane0.ptr = NULL;
	plane1.ptr = NULL;
	plane2.ptr = NULL;

	of_bits.close();
#ifdef 	OutputFile	
	fclose(fp);
#endif	
	return 0;
}
