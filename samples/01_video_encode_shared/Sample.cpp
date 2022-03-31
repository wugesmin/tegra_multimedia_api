#include "Nv_Tegra_Enc.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

FILE *pFile = 0;
FILE *pOutputFile = 0;    
unsigned char *pSrcData = 0;

int main(int argc, char *argv[])
{
    int hr = 0;
    NVTegraEnc pDevice = 0;
    unsigned int nFrameCount = 0;

    pFile = fopen("test.yuv", "r");
    pOutputFile = fopen("OutputA", "wb");
    pSrcData = (unsigned char *)malloc(10000000);

    hr = NVTEGRAENC_CREATE_ENCODER(&pDevice);
    //cout << "Create encoder hr = " << hr << endl;
	
    hr = NVTEGRAENC_SET_ENCODER_FORMAT(pDevice, NVTEGRAENC_TYPE_H264, NVTEGRAENC_FORMAT_YUY2, 1920, 1080, 30, NVTEGRAENC_RATECONTROL_CBR, 
                                       16000000, NVTEGRAENC_PRESET_ULTRAFAST/*NVTEGRAENC_PRESET_SLOW*/, 30, 
		                       NVTEGRAENC_PROFILE_H264_MAIN, NVTEGRAENC_LEVEL_H264_5_1, 0);
    //cout << "Set encoder hr = " << hr << endl;
    
    unsigned int nDataRead = 1920 * 1080 * 2;
    size_t nReadSize = 0; 
    uint8_t *pDestBuffer = 0;
    ULONG pDestBufferSize = 0;
    bool pbIsKeyFrame = 0;
    unsigned int nOutputFrameCount = 0;

    while(!feof(pFile))
    {
        nReadSize = fread(pSrcData, 1, nDataRead, pFile);
        
        pDestBuffer = 0;
        pDestBufferSize = 0;
        pbIsKeyFrame = 0;

        if(nReadSize)
        {
	    //Set key frame test, use pbIsKeyFrame to verify whether the output frame is key frame after calling this function.
	    //There's a latency(some non-keyframes) between calling this function and the output of required key frame.
	    //if(nFrameCount == 5)	
		//hr = NVTEGRAENC_SET_KEYFRAME(pDevice);	

            hr = NVTEGRAENC_FRAME_ENCODE(pDevice, pSrcData, &pDestBuffer, &pDestBufferSize, &pbIsKeyFrame);

            //cout << "Encode hr = " << hr << " Count = " << ++nFrameCount << " Data size = " << pDestBufferSize << endl;

            if(pDestBuffer && pDestBufferSize)
            {
                nOutputFrameCount++;

		//cout << "Write data size = " << pDestBufferSize << " FrameNum = " << nOutputFrameCount << " key frame = " << pbIsKeyFrame << endl;

                fwrite(pDestBuffer, 1, pDestBufferSize, pOutputFile);
          
	    }
        }
	else
	    break;
    }

    while(1)	//Get remaining data with NULL pSrcData.
    {
        pDestBuffer = 0;
        pDestBufferSize = 0;
        pbIsKeyFrame = 0;

        hr = NVTEGRAENC_FRAME_ENCODE(pDevice, NULL, &pDestBuffer, &pDestBufferSize, &pbIsKeyFrame);
            
        //cout << "Encode hr = " << hr << " Count = " << ++nFrameCount << " Data size = " << pDestBufferSize << endl;

        if(pDestBuffer && pDestBufferSize)
        {
            nOutputFrameCount++;
	    //cout << "Write data size = " << pDestBufferSize << " FrameNum = " << nOutputFrameCount << " key frame = " << pbIsKeyFrame << endl;

            fwrite(pDestBuffer, 1, pDestBufferSize, pOutputFile);    
	}
   	else
	    break;	
    }

    hr = NVTEGRAENC_DESTROY_ENCODER(pDevice);

    fclose(pFile);
    fclose(pOutputFile);
    free(pSrcData);

    return hr;

}
