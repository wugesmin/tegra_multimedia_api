#ifndef _NV_TEGRA_CONVERT_H_
#define _NV_TEGRA_CONVERT_H_

#include <stdint.h>

enum INPUT_COLOR_FORMAT
{
	INPUT_COLOR_FORMAT_YV12 = 0,
	INPUT_COLOR_FORMAT_YUY2 = 1,
	INPUT_COLOR_FORMAT_NV12 = 2,
	INPUT_COLOR_FORMAT_ABGR = 3,
	INPUT_COLOR_FORMAT_XRGB = 4,
	INPUT_COLOR_FORMAT_RGB = 5,
	INPUT_COLOR_FORMAT_BGR = 6,
	INPUT_COLOR_FORMAT_ARGB = 7
};

typedef void* NVTegraConvert;
typedef bool (*NVTegraConvert_Callback)(void *pDataBuffer, unsigned int nDataSize, void *pUserData);

int NVTEGRACONVERT_CREATE(NVTegraConvert *device);
int NVTEGRACONVERT_DESTROY(NVTegraConvert device);
int NVTEGRACONVERT_SET_FORMAT(NVTegraConvert device, INPUT_COLOR_FORMAT nColorFormat, unsigned int nInputWidth, unsigned int nInputHeight, unsigned int nOutputWidth, unsigned int nOutputHeight, unsigned int nCropLeft, unsigned int nCropTop, unsigned int nCropWidth, unsigned int nCropHeight);
int NVTEGRACONVERT_FRAME_CONVERT(NVTegraConvert device, uint8_t *pSrcBuffer, unsigned int nSrcBufferSize);
int NVTEGRACONVERT_SET_CALLBACK(NVTegraConvert device, NVTegraConvert_Callback cb, void *pUserData);

#endif	//_NV_TEGRA_CONVERT_H_
