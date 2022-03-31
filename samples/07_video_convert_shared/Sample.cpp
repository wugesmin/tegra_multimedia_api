#include "Nv_Tegra_Convert.h"
#include <iostream>
#include <cuda_runtime_api.h>
//#include <unistd.h>
//#include <stdio.h>
//#include <stdlib.h>

using namespace std;
unsigned char *pData = NULL;
unsigned int gnInWidth = 1920;
unsigned int gnInHeight = 1080;
unsigned int gnOutWidth = 608;
unsigned int gnOutHeight = 608;

bool GetConvertData(void *pDataBuffer, unsigned int nDataSize, void *pUserData)
{
    bool ret = 0;
	int x = gnOutWidth * 19;
    float *pfloatData = NULL;    
#if(1)
    if(!pData)
        pData = (unsigned char *)malloc(gnOutWidth * gnOutHeight * 3 * 4);

    cudaMemcpy(pData, pDataBuffer, nDataSize, cudaMemcpyDeviceToHost);

    pfloatData = (float *)pData;
    printf(" %f, %f",  pfloatData[0], pfloatData[x]);	// B
    printf(" %f, %f",  *(pfloatData + gnOutWidth * gnOutHeight), *(pfloatData + gnOutWidth * gnOutHeight + x));	// G
    printf(" %f, %f\n",  *(pfloatData + gnOutWidth * gnOutHeight * 2), *(pfloatData + gnOutWidth * gnOutHeight * 2 + x));	// R
#endif
    if(nDataSize)    
	ret = 1;

    return ret;
}

int main(int argc, char *argv[])
{
	int hr = 0;
	NVTegraConvert pDevice = 0;
	FILE *pInputFile = NULL;
	pInputFile = fopen("test_nv12.yuv", "rb");
    uint8_t *pSrcBuffer = NULL;
    unsigned int nSrcBufferSize = 0;
	unsigned int nFrameCount = 0;
	unsigned int nCount = 0;
	unsigned int nCountA = 0;

	pSrcBuffer = (uint8_t *)malloc(gnInWidth * gnInHeight * 3); 

	while(nCount++ < 1)
	{
		fseek(pInputFile, 0, SEEK_SET);

		NVTegraConvert_Callback pCB = &GetConvertData;

		//printf("Count = %d\n", nCount);
		hr = NVTEGRACONVERT_CREATE(&pDevice);
		//cout << "NVTEGRACONVERT_CREATE " << hr << " " << pDevice << endl;
	
		hr = NVTEGRACONVERT_SET_FORMAT(pDevice, INPUT_COLOR_FORMAT_NV12, gnInWidth, gnInHeight, gnOutWidth, gnOutHeight, 0, 0, 0, 0);
		//cout << "NVTEGRACONVERT_SET_FORMAT " << hr << endl;

		hr = NVTEGRACONVERT_SET_CALLBACK(pDevice, pCB, NULL);
		//cout << "NVTEGRACONVERT_SET_CALLBACK " << hr << endl;

		unsigned int nSize = gnInWidth * gnInHeight * 3/2;
		//unsigned int nSize = 1920 * 1080 * 2;
		unsigned int nReadSize = 0;
		while(!feof(pInputFile))
		{
			nReadSize = fread(pSrcBuffer, 1, nSize, pInputFile);
			if(nReadSize)
			{
				//cout << "NVTEGRACONVERT_FRAME_CONVERT Begin" << endl;
				hr = NVTEGRACONVERT_FRAME_CONVERT(pDevice, pSrcBuffer, nReadSize);
				//cout << "NVTEGRACONVERT_FRAME_CONVERT " << hr << " Count " << ++nFrameCount << endl;
			}		
			else 
			{
				if(nCountA++ < 0)
					fseek(pInputFile, 0, SEEK_SET);
				else
					break;
			}
		}

		hr = NVTEGRACONVERT_DESTROY(pDevice);
		//cout << "NVTEGRACONVERT_DESTROY " << hr << endl;
	}

	fclose(pInputFile);
	free(pSrcBuffer);
    if(pData)
        free(pData);

	return 0;
}
