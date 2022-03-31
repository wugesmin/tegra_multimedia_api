#ifndef _ENCODE_DEVICE_H_
#define _ENCODE_DEVICE_H_

#include "video_encode.h"
#include <queue>
#include <vector>

typedef unsigned long ULONG;

struct EncodeParams
{
    ULONG nType;
    ULONG nWidth;
    ULONG nHeight;
    double dFrameRate;
    ULONG nRecordMode;
    ULONG nBitRate;
    ULONG nTargetUsage;
    ULONG nGOP;
    ULONG nCodecProfile;
    ULONG nCodecLevel;
    ULONG nBFrameNumber;
    ULONG nInputFormat;
};

struct FrameInfo
{
    uint8_t *pData;
    ULONG nDataSize;
    ULONG nMaxDataSize;
    bool bIsKeyFrame;
    bool bIsUsed;
};

class CNvTegraEncode
{
public:
    CNvTegraEncode();
    ~CNvTegraEncode();

    std::queue<FrameInfo *> m_FrameInfoQueue;
    std::vector<FrameInfo> m_FrameInfoArray; 
    pthread_mutex_t m_OutputMutex;
    bool m_bFirstOutput;

    int Create();
    void Release();
    int SetFormat(EncodeParams nParams);
    int EncodeFrame(uint8_t *pSrcBuffer, uint8_t **pDestBuffer, ULONG *pDestBufferSize, bool *pbIsKeyFrame);
    int InsertKeyFrame();
    int GetFreeIndex();
    void ResizeBuffer(FrameInfo *pFrameInfo, ULONG nSize);

	uint8_t *pConverterBuffer;
	uint8_t *pInputBuffer;
	uint8_t *pOutputBuffer;
    unsigned int iInputSize;
    unsigned int iConvertSize;
private:
    context_t m_nCtx;
    unsigned int m_nFrameCount;
    unsigned int m_nQueueIndex;    
    bool m_bIsEndOfEncode;
    unsigned int m_nChangeFormat;		// 0:dont change 1:YV12 to I420 2: YUY2 to I420    
    void FrameRateConvert(double dFrameRate, unsigned int *pFrameRateNum, unsigned int *pFrameRateDen);
};

#endif
