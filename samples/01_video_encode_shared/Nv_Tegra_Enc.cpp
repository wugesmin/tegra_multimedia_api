#include "Nv_Tegra_Enc.h"
#include "Encode_Device.h"

#define EXPORT __attribute__((visibility("default")))

EXPORT int NVTEGRAENC_CREATE_ENCODER(NVTegraEnc *device)
{
    if(!device)
        return -1;    

    int hr = 0;
    
    CNvTegraEncode *pDevice = NULL;
    pDevice = new CNvTegraEncode;

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

EXPORT int NVTEGRAENC_DESTROY_ENCODER(NVTegraEnc device)
{
    if(!device)
        return -1;  

    CNvTegraEncode *pDevice = (CNvTegraEncode *)device;
    pDevice->Release();
    delete pDevice;

    return 0;
}

EXPORT int NVTEGRAENC_SET_ENCODER_FORMAT(NVTegraEnc device, ULONG nType, ULONG nInputFormat, ULONG nWidth, ULONG nHeight, double dFrameRate, ULONG nRecordMode, ULONG nBitRate, ULONG nTargetUsage, ULONG nGOP, ULONG nCodecProfile, ULONG nCodecLevel, ULONG nBFrameNumber)
{
    if(!device)
        return -1;  

    int hr = 0;

    CNvTegraEncode *pDevice = (CNvTegraEncode *)device;
    EncodeParams nParams;
    
    memset(&nParams, 0, sizeof(nParams));
    nParams.nType = nType;
    nParams.nWidth = nWidth;
    nParams.nHeight = nHeight;
    nParams.dFrameRate = dFrameRate;
    nParams.nRecordMode = nRecordMode;
    nParams.nBitRate = nBitRate;
    nParams.nTargetUsage = nTargetUsage;
    nParams.nGOP = nGOP;
    nParams.nCodecProfile = nCodecProfile;
    nParams.nCodecLevel = nCodecLevel;
    nParams.nBFrameNumber = nBFrameNumber;
    nParams.nInputFormat = nInputFormat;
    hr = pDevice->SetFormat(nParams);

    return hr;
}

EXPORT int NVTEGRAENC_FRAME_ENCODE(NVTegraEnc device, uint8_t *pSrcBuffer, uint8_t **pDestBuffer, ULONG *pDestBufferSize, bool *pbIsKeyFrame)
{
    if(!device)
        return -1;  

    int hr = 0;

    CNvTegraEncode *pDevice = (CNvTegraEncode *)device;
    hr = pDevice->EncodeFrame(pSrcBuffer, pDestBuffer, pDestBufferSize, pbIsKeyFrame);

    return hr;
}

int NVTEGRAENC_SET_KEYFRAME(NVTegraEnc device)
{
    if(!device)
        return -1;  

    int hr = 0;

    CNvTegraEncode *pDevice = (CNvTegraEncode *)device;
    hr = pDevice->InsertKeyFrame();
    
    return hr;
}
