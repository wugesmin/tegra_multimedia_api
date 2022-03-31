#ifndef _CONVERT_DEVICE_H_
#define _CONVERT_DEVICE_H_

#include "video_convert.h"
#include "Nv_Tegra_Convert.h"

class CNvTegraConvert
{
public:
	CNvTegraConvert();
	~CNvTegraConvert();

	NVTegraConvert_Callback m_fpCallBack;
	unsigned int m_nCudaBufferSize;
	unsigned int m_nChangeFormat;	// 0:dont change 1:using RGB format 2:using BGR format 3:using ARGB
	void *m_pUserData;

	int Create();
	void Release();
	int SetFormat(INPUT_COLOR_FORMAT nColorFormat, unsigned int nInputWidth, unsigned int nInputHeight, unsigned int nOutputWidth, unsigned int nOutputHeight, unsigned int nCropLeft, unsigned int nCropTop, unsigned int nCropWidth, unsigned int nCropHeight);
	int ConvertFrame(uint8_t *pSrcBuffer, unsigned int nSrcBufferSize);
	void SetCallBack(NVTegraConvert_Callback cb, void *pUserData);	
	void abort(context_t * ctx);
	
private:
	context_t ctx;
	unsigned int m_nFrameCount;
	
};

#endif	//_CONVERT_DEVICE_H_

