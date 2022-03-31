#include "Nv_Tegra_Convert.h"
#include "Convert_Device.h"

#define EXPORT __attribute__((visibility("default")))

EXPORT int NVTEGRACONVERT_CREATE(NVTegraConvert *device)
{   
	if(!device)
		return -1;  

	int hr = 0;

	CNvTegraConvert *pDevice = NULL;
	pDevice = new CNvTegraConvert;

    if(pDevice)
    {
        hr = pDevice->Create();
        if(hr != 0)
        {
            pDevice->Release();
            delete pDevice;
            *device = NULL;
            return hr;
        }
    }
    else
    {
        *device = NULL;
        return -1;
    }

    *device = pDevice;	

	return hr;
}

EXPORT int NVTEGRACONVERT_DESTROY(NVTegraConvert device)
{
    if(!device)
        return -1;  

    CNvTegraConvert *pDevice = (CNvTegraConvert *)device;
    pDevice->Release();
    delete pDevice;

	return 0;
}

EXPORT int NVTEGRACONVERT_SET_FORMAT(NVTegraConvert device, INPUT_COLOR_FORMAT nColorFormat, unsigned int nInputWidth, unsigned int nInputHeight, unsigned int nOutputWidth, unsigned int nOutputHeight, unsigned int nCropLeft, unsigned int nCropTop, unsigned int nCropWidth, unsigned int nCropHeight)
{
    if(!device)
        return -1;  

    int hr = 0;

    CNvTegraConvert *pDevice = (CNvTegraConvert *)device;
    hr = pDevice->SetFormat(nColorFormat, nInputWidth, nInputHeight, nOutputWidth, nOutputHeight, nCropLeft, nCropTop, nCropWidth, nCropHeight);
    return hr;
}

EXPORT int NVTEGRACONVERT_FRAME_CONVERT(NVTegraConvert device, uint8_t *pSrcBuffer, unsigned int nSrcBufferSize)
{
    if(!device)
        return -1;  

    int hr = 0;

    CNvTegraConvert *pDevice = (CNvTegraConvert *)device;
	hr = pDevice->ConvertFrame(pSrcBuffer, nSrcBufferSize);

	return hr;
}

EXPORT int NVTEGRACONVERT_SET_CALLBACK(NVTegraConvert device, NVTegraConvert_Callback cb, void *pUserData)
{
	int hr = 0;

    CNvTegraConvert *pDevice = (CNvTegraConvert *)device;
	pDevice->SetCallBack(cb, pUserData);

	return hr;
}
