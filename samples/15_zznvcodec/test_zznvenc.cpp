#include "zznvcodec.h"
#include "ZzLog.h"
#include <unistd.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <npp.h>
#include <cuda_runtime.h>

ZZ_INIT_LOG("test_zznvenc");

uint8_t* zziMalloc_8u_C1(int width, int height, int* step) {
	*step = width;
	return (uint8_t*)aligned_alloc(16, width * height);
}

void zziFree(void* p) {
	free(p);
}

void _zznvcodec_encoder_on_video_packet(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp, intptr_t pUser) {
	LOGD("pBuffer=%p, nSize=%d, nFlags=%d, nTimestamp=%.2f", pBuffer, nSize, nFlags, nTimestamp / 1000.0);
}

int main(int argc, char *argv[])
{
	zznvcodec_encoder_t* pEnc = zznvcodec_encoder_new();
	zznvcodec_encoder_set_video_property(pEnc, 1920, 1080, ZZNVCODEC_PIXEL_FORMAT_YUV420P);
	int nEncoderPixFmt = ZZNVCODEC_PIXEL_FORMAT_H264;
	zznvcodec_encoder_set_misc_property(pEnc, ZZNVCODEC_PROP_ENCODER_PIX_FMT, (intptr_t)&nEncoderPixFmt);
	zznvcodec_encoder_register_callbacks(pEnc, _zznvcodec_encoder_on_video_packet, (intptr_t)0);
	zznvcodec_encoder_start(pEnc);

	zznvcodec_video_frame_t oVideoFrame;
	oVideoFrame.num_planes = 3;

	zznvcodec_video_plane_t& plane0 = oVideoFrame.planes[0];
	plane0.width = 1920;
	plane0.height = 1080;
#if 1 // USE_CUDA_MEMORY
	plane0.ptr = nppiMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);
#else
	plane0.ptr = zziMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);
#endif
	LOGI("plane0.ptr = %p / %d", plane0.ptr, plane0.stride);

	zznvcodec_video_plane_t& plane1 = oVideoFrame.planes[1];
	plane1.width = 1920 / 2;
	plane1.height = 1080 / 2;
#if 1 // USE_CUDA_MEMORY
	plane1.ptr = nppiMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);
#else
	plane1.ptr = zziMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);
#endif
	LOGI("plane1.ptr = %p / %d", plane1.ptr, plane1.stride);

	zznvcodec_video_plane_t& plane2 = oVideoFrame.planes[2];
	plane2.width = 1920 / 2;
	plane2.height = 1080 / 2;
#if 1 // USE_CUDA_MEMORY
	plane2.ptr = nppiMalloc_8u_C1(plane2.width, plane2.height, &plane2.stride);
#else
	plane2.ptr = zziMalloc_8u_C1(plane2.width, plane2.height, &plane2.stride);
#endif
	LOGI("plane2.ptr = %p / %d", plane2.ptr, plane2.stride);

	for(int i = 0;i < 256;++i) {
		LOGI("Frame %d", i);
		zznvcodec_encoder_set_video_uncompression_buffer(pEnc, &oVideoFrame, 1 * 1000000L);
	}

	zznvcodec_encoder_stop(pEnc);

	zznvcodec_encoder_delete(pEnc);
	pEnc = NULL;

#if 1 // USE_CUDA_MEMORY
	nppiFree(plane0.ptr);
	nppiFree(plane1.ptr);
	nppiFree(plane2.ptr);
#else
	zziFree(plane0.ptr);
	zziFree(plane1.ptr);
	zziFree(plane2.ptr);
#endif

	return 0;
}
