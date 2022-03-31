#ifndef _NV_TEGRA_ENC_H_
#define _NV_TEGRA_ENC_H_

#include <stdint.h>

#define NVTEGRAENC_TYPE_H264 0
#define NVTEGRAENC_TYPE_H265 1

#define NVTEGRAENC_RATECONTROL_VBR 0
#define NVTEGRAENC_RATECONTROL_CBR 1

#define NVTEGRAENC_PROFILE_H264_BASELINE 0
#define NVTEGRAENC_PROFILE_H264_MAIN     2
#define NVTEGRAENC_PROFILE_H264_HIGH     4

#define NVTEGRAENC_PROFILE_H265_MAIN 0

#define NVTEGRAENC_LEVEL_H264_1_0 0 
#define NVTEGRAENC_LEVEL_H264_1b  1
#define NVTEGRAENC_LEVEL_H264_1_1 2
#define NVTEGRAENC_LEVEL_H264_1_2 3 
#define NVTEGRAENC_LEVEL_H264_1_3 4
#define NVTEGRAENC_LEVEL_H264_2_0 5
#define NVTEGRAENC_LEVEL_H264_2_1 6
#define NVTEGRAENC_LEVEL_H264_2_2 7
#define NVTEGRAENC_LEVEL_H264_3_0 8
#define NVTEGRAENC_LEVEL_H264_3_1 9
#define NVTEGRAENC_LEVEL_H264_3_2 10
#define NVTEGRAENC_LEVEL_H264_4_0 11
#define NVTEGRAENC_LEVEL_H264_4_1 12
#define NVTEGRAENC_LEVEL_H264_4_2 13
#define NVTEGRAENC_LEVEL_H264_5_0 14
#define NVTEGRAENC_LEVEL_H264_5_1 15
//Note : Currently, not support level setting for H265. 
//It doesn't work to set nCodecLevel on NVTEGRAENC_SET_ENCODER_FORMAT() for H265.
//Note : Currently, B frame setting is not support for H.265.
//It doesn't work to set nBFrameNumber larger than 0 on NVTEGRAENC_SET_ENCODER_FORMAT() for H265.

#define NVTEGRAENC_PRESET_ULTRAFAST 1
#define NVTEGRAENC_PRESET_FAST 	    2
#define NVTEGRAENC_PRESET_MEDIUM    3
#define NVTEGRAENC_PRESET_SLOW      4

#define NVTEGRAENC_FORMAT_YV12 0
#define NVTEGRAENC_FORMAT_YUY2 1
#define NVTEGRAENC_FORMAT_I420 2
typedef unsigned long ULONG;
typedef void* NVTegraEnc;

int NVTEGRAENC_CREATE_ENCODER(NVTegraEnc *device);
int NVTEGRAENC_DESTROY_ENCODER(NVTegraEnc device);
int NVTEGRAENC_SET_ENCODER_FORMAT(NVTegraEnc device, ULONG nType, ULONG nInputFormat, ULONG nWidth, ULONG nHeight, double dFrameRate, ULONG nRecordMode, ULONG nBitRate, ULONG nTargetUsage, ULONG nGOP, ULONG nCodecProfile, ULONG nCodecLevel, ULONG nBFrameNumber);
int NVTEGRAENC_FRAME_ENCODE(NVTegraEnc device, uint8_t *pSrcBuffer, uint8_t **pDestBuffer, ULONG *pDestBufferSize, bool *pbIsKeyFrame);
int NVTEGRAENC_SET_KEYFRAME(NVTegraEnc device);	
//Note : There's a latency(some non-keyframes) between calling of this function and the output of required key frame. And do NOT sure which frame would be the key frame after calling this function. 

#endif	//_NV_TEGRA_ENC_H_
