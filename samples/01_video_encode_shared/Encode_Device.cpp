#include "Encode_Device.h"
#include "Nv_Tegra_Enc.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include "../common/algorithm/cuda/NvCudaProc.h"
#define DEBUG_INFO 0

using namespace std;

static bool
encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer,
                                  NvBuffer * shared_buffer, void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvVideoEncoder *enc = ctx->enc;
    CNvTegraEncode *pDevice = (CNvTegraEncode *)ctx->pEncodeDevice;

    uint32_t frame_num = ctx->enc->capture_plane.getTotalDequeuedBuffers() - 1;
    uint32_t ReconRef_Y_CRC = 0;
    uint32_t ReconRef_U_CRC = 0;
    uint32_t ReconRef_V_CRC = 0;
    //static uint32_t num_encoded_frames = 1;	//test
#if(DEBUG_INFO)
    static uint32_t num_encoded_frames = 0;	//test
#endif

    if (v4l2_buf == NULL)
    {
        ctx->got_error = true;
        ctx->enc->abort();
        return false;
    }

    // GOT EOS from encoder. Stop dqthread.
    if (buffer->planes[0].bytesused == 0)
    {
        return false;
    }

    //test
    //cout << "[TEST] Write output begin frame_num = " << frame_num << endl;

    //fwrite(buffer->planes[0].data, 1, buffer->planes[0].bytesused, ctx->pOutputFile);

    v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
    enc->getMetadata(v4l2_buf->index, enc_metadata);	

    pthread_mutex_lock(&pDevice->m_OutputMutex);
#if(DEBUG_INFO)
    cout << "Thread Output begin"<< endl;
#endif
    int nIndex = 0;
    nIndex = pDevice->GetFreeIndex();

    //cout << "Get output index = " << nIndex << endl;

    if(pDevice->m_FrameInfoArray[nIndex].nMaxDataSize < buffer->planes[0].bytesused)
        pDevice->ResizeBuffer(&pDevice->m_FrameInfoArray[nIndex], buffer->planes[0].bytesused);

    memcpy(pDevice->m_FrameInfoArray[nIndex].pData, buffer->planes[0].data, buffer->planes[0].bytesused);
    pDevice->m_FrameInfoArray[nIndex].nDataSize = buffer->planes[0].bytesused;
    pDevice->m_FrameInfoArray[nIndex].bIsKeyFrame = enc_metadata.KeyFrame;
    pDevice->m_FrameInfoArray[nIndex].bIsUsed = true;

    pDevice->m_FrameInfoQueue.push(&pDevice->m_FrameInfoArray[nIndex]);
#if(DEBUG_INFO)
    cout << "Thread Output end"<< endl;
#endif    
    pthread_mutex_unlock(&pDevice->m_OutputMutex);

    //write_encoder_output_frame(ctx->out_file, buffer);
#if(DEBUG_INFO)
    num_encoded_frames++;

    //test
    //cout << "Write output end Count = " << num_encoded_frames << endl;
#endif

    if (enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0)
    {
        ctx->got_error = true;
        ctx->enc->abort();
        return false;
    }

    return true;
}

CNvTegraEncode::CNvTegraEncode() :
    m_nFrameCount(0),
    m_nQueueIndex(0),
    m_bFirstOutput(true),
    m_bIsEndOfEncode(false)
{
    memset(&m_nCtx, 0, sizeof(m_nCtx));
}

CNvTegraEncode::~CNvTegraEncode()
{
}

int CNvTegraEncode::Create()
{
    m_nCtx.enc = NvVideoEncoder::createVideoEncoder("enc0");
    //cout << "enc = " << m_nCtx.enc << endl;
    if(!m_nCtx.enc)
        return -1;

    //test
    //m_nCtx.pOutputFile = fopen("Output", "w");
    //if(!m_nCtx.pOutputFile)
    //    return -1;

    m_FrameInfoArray.resize(10);
    for(unsigned int i = 0 ; i < m_FrameInfoArray.size() ; i++)
    {
        memset(&m_FrameInfoArray[i], 0, sizeof(FrameInfo));
        m_FrameInfoArray[i].nMaxDataSize = 100;
        m_FrameInfoArray[i].pData = new unsigned char[100];
    }
    
    m_OutputMutex = PTHREAD_MUTEX_INITIALIZER;

    return 0;
}

void CNvTegraEncode::Release()
{
    if(m_nCtx.enc)
    {
        delete m_nCtx.enc;
        m_nCtx.enc = NULL;
    }

	if(m_nChangeFormat == 2)
	{
        cudaFree(pInputBuffer);
        cudaFree(pOutputBuffer);
		free(pConverterBuffer);
	}
    //test
    //if(m_nCtx.pOutputFile)
    //{
    //    fclose(m_nCtx.pOutputFile);
    //    m_nCtx.pOutputFile = NULL;
    //}

    for(unsigned int i = 0 ; i < m_FrameInfoArray.size() ; i++)
        delete [] m_FrameInfoArray[i].pData;

    if(m_FrameInfoArray.size())
        m_FrameInfoArray.clear();

}

int CNvTegraEncode::SetFormat(EncodeParams nParams)
{
    int hr = 0;

    if(nParams.nType == NVTEGRAENC_TYPE_H264)
        m_nCtx.encoder_pixfmt = V4L2_PIX_FMT_H264;	
    else if(nParams.nType == NVTEGRAENC_TYPE_H265)
	m_nCtx.encoder_pixfmt = V4L2_PIX_FMT_H265;	
    else
	    return -1;

    m_nCtx.width = nParams.nWidth;
    m_nCtx.height = nParams.nHeight;

    if(nParams.nInputFormat == NVTEGRAENC_FORMAT_YV12)
        m_nChangeFormat = 1;	
    else if(nParams.nInputFormat == NVTEGRAENC_FORMAT_YUY2)
	{
	    m_nChangeFormat = 2;		        
        iInputSize = m_nCtx.width * m_nCtx.height * 2;
        iConvertSize = m_nCtx.width * m_nCtx.height * 3 / 2;
        pConverterBuffer = (uint8_t*)malloc(iConvertSize);
        cudaMalloc((void**) &pInputBuffer, iInputSize);
        cudaMalloc((void**) &pOutputBuffer, iConvertSize);
	}

    unsigned int nFrameRateNum = 0;
    unsigned int nFrameRateDen = 0;
    FrameRateConvert(nParams.dFrameRate, &nFrameRateNum, &nFrameRateDen);
    m_nCtx.fps_n = nFrameRateNum;
    m_nCtx.fps_d = nFrameRateDen;
    m_nCtx.ratecontrol = (enum v4l2_mpeg_video_bitrate_mode)nParams.nRecordMode;
    m_nCtx.bitrate = nParams.nBitRate;
    m_nCtx.hw_preset_type = (enum v4l2_enc_hw_preset_type)nParams.nTargetUsage;
    m_nCtx.iframe_interval = nParams.nGOP;
    m_nCtx.idr_interval = nParams.nGOP;
    m_nCtx.profile = (enum v4l2_mpeg_video_h264_profile)nParams.nCodecProfile;
    m_nCtx.level = (enum v4l2_mpeg_video_h264_level)nParams.nCodecLevel;
    m_nCtx.num_b_frames = nParams.nBFrameNumber;

    m_nCtx.insert_sps_pps_at_idr = true;
    m_nCtx.insert_aud = true;
    m_nCtx.insert_vui = true;

    m_nCtx.pEncodeDevice = (void *)this;

    hr = m_nCtx.enc->setCapturePlaneFormat(m_nCtx.encoder_pixfmt, m_nCtx.width, m_nCtx.height, 2 * 1024 * 1024);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, m_nCtx.width, m_nCtx.height);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setFrameRate(m_nCtx.fps_n, m_nCtx.fps_d);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setRateControlMode(m_nCtx.ratecontrol);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setBitrate(m_nCtx.bitrate);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setHWPresetType(m_nCtx.hw_preset_type);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setIDRInterval(m_nCtx.idr_interval);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setIFrameInterval(m_nCtx.iframe_interval);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setProfile(m_nCtx.profile);
    if(hr != 0)
        return hr;

    //Note : Currently, there's no level setting for H265 according to NvVideoEncoder.cpp.
    hr = m_nCtx.enc->setLevel(m_nCtx.level);
    if(hr != 0)
        return hr;

    if(m_nCtx.num_b_frames)
    {
    	hr = m_nCtx.enc->setNumBFrames(m_nCtx.num_b_frames);
    	if(hr != 0)
    	    return hr;
    }

    hr = m_nCtx.enc->setInsertSpsPpsAtIdrEnabled(m_nCtx.insert_sps_pps_at_idr);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setInsertVuiEnabled(m_nCtx.insert_vui);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->setInsertAudEnabled(m_nCtx.insert_aud);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->output_plane.setStreamStatus(true);
    if(hr != 0)
        return hr;

    hr = m_nCtx.enc->capture_plane.setStreamStatus(true);
    if(hr != 0)
        return hr;

    m_nCtx.enc->capture_plane.
        setDQThreadCallback(encoder_capture_plane_dq_callback);

    m_nCtx.enc->capture_plane.startDQThread(&m_nCtx);

    for (uint32_t i = 0; i < m_nCtx.enc->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        hr = m_nCtx.enc->capture_plane.qBuffer(v4l2_buf, NULL);
        if (hr < 0) //Queue capture buffer error
        {
            m_nCtx.got_error = true;
            m_nCtx.enc->abort();
            return hr;
        }
    }

    return 0;
}

int CNvTegraEncode::EncodeFrame(uint8_t *pSrcBuffer, uint8_t **pDestBuffer, ULONG *pDestBufferSize, bool *pbIsKeyFrame)
{
    int hr = 0;

    if(!m_bIsEndOfEncode)
    {
        m_nFrameCount++;

        struct v4l2_buffer v4l2_buf;
    	struct v4l2_plane planes[MAX_PLANES];
    	NvBuffer *buffer = NULL;

    	memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    	memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    	if(m_nFrameCount <= m_nCtx.enc->output_plane.getNumBuffers())
            v4l2_buf.index = m_nFrameCount - 1;

    	v4l2_buf.m.planes = planes;

    	if(m_nFrameCount <= m_nCtx.enc->output_plane.getNumBuffers())
            buffer = m_nCtx.enc->output_plane.getNthBuffer(m_nFrameCount - 1);
    	else if (m_nCtx.enc->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10) < 0)
    	{
            m_nCtx.got_error = true;
       	    m_nCtx.enc->abort();
            return -1;
    	}

    	NvBuffer::NvBufferPlane *plane = NULL;
    	uint8_t *pSrcData = pSrcBuffer;
    	char *data = NULL;

    	if(pSrcBuffer)
    	{    
            // color space conversion    
#ifdef CPU_Covert             
            if(m_nChangeFormat == 2)
            {	
               
				unsigned int Ycount, Ucount, Vcount = 0;
				NvBuffer::NvBufferPlane *Uplane = &(buffer->planes[1]);
				char *Udata = (char *)Uplane->data;
				Uplane->bytesused = 0;
				NvBuffer::NvBufferPlane *Vplane = &(buffer->planes[2]);
				char *Vdata = (char *)Vplane->data;
				Vplane->bytesused = 0;
				plane = &(buffer->planes[0]); 
	            data = (char *)plane->data;
	            plane->bytesused = 0;
				unsigned int readwidth = plane->fmt.width << 1;

                for (unsigned int j = 0 ; j < plane->fmt.height ; j++)
                {
					Ycount = Ucount = Vcount = 0;

					if(j%2 == 0)
					{
		                for(unsigned int k = 0 ; k < readwidth ; k+=4)
		                {
		                    data[Ycount++] = pSrcData[j*readwidth+k];
							Udata[Ucount++] = pSrcData[j*readwidth+k+1];
							data[Ycount++] = pSrcData[j*readwidth+k+2];
							Vdata[Vcount++] = pSrcData[j*readwidth+k+3];
		                }
						data += plane->fmt.stride;
						Udata += Uplane->fmt.stride;
						Vdata += Vplane->fmt.stride;					
					}
					else
					{
		                for(unsigned int k = 0 ; k < readwidth ; k+=2)
		                {
		                    data[Ycount++] = pSrcData[j*readwidth+k];
		                }
						data += plane->fmt.stride;
					}
                } 
				plane->bytesused = plane->fmt.stride * plane->fmt.height;
				Uplane->bytesused = Uplane->fmt.stride * Uplane->fmt.height;
				Vplane->bytesused = Vplane->fmt.stride * Vplane->fmt.height;
            }
#endif
            if(m_nChangeFormat == 2)
            {
                cudaMemcpy(pInputBuffer,pSrcBuffer, iInputSize, cudaMemcpyHostToDevice);			
                if(m_nChangeFormat == 2) //  YUY2 to I420
                {
                    colorConvert((void*)pInputBuffer,
                        m_nCtx.width*2,
                        m_nCtx.height,
                        COLOR_FORMAT_YUY2TOI420,
                        (char *)pOutputBuffer);
                }
                cudaMemcpy(pConverterBuffer,pOutputBuffer,iConvertSize,cudaMemcpyDeviceToHost);                
                pSrcData = pConverterBuffer;
            }

            for(unsigned int i = 0 ; i < buffer->n_planes ; i++)
            {
                plane = &(buffer->planes[i]);
                unsigned int readwidth = plane->fmt.bytesperpixel * plane->fmt.width;
                data = (char *)plane->data;
                plane->bytesused = 0;

                if(m_nChangeFormat == 1)
                {
                    if(i == 1)
                        plane = &(buffer->planes[2]);
                    else if(i == 2)   
                        plane = &(buffer->planes[1]);  
                }

                for (unsigned int j = 0 ; j < plane->fmt.height ; j++)
                {
                    memcpy(data, pSrcData, readwidth);
                    pSrcData += readwidth;
                    data += plane->fmt.stride;
                }

                plane->bytesused = plane->fmt.stride * plane->fmt.height;
            }	
    	}
    	else
        {
            m_bIsEndOfEncode = true;
            v4l2_buf.m.planes[0].bytesused = 0;
        }

        hr = m_nCtx.enc->output_plane.qBuffer(v4l2_buf, NULL);

        if (hr < 0)
        {
            m_nCtx.got_error = true;
            m_nCtx.enc->abort();
            return hr;
        }
        if(m_bIsEndOfEncode)
            m_nCtx.enc->capture_plane.waitForDQThread(-1);
    }
    
    pthread_mutex_lock(&m_OutputMutex);

#if(DEBUG_INFO)
    cout << "Thread Encode output begin"<< endl;
#endif
    //if(!m_bFirstOutput && !m_FrameInfoQueue.size())      
    //if(m_bFirstOutput)
    //	cout << "Thread Encode output Total Frame Num in Queue = " << m_FrameInfoQueue.size() << endl;  
    
    //begin to output frame when there're more than 3 frames in the queue for sync between encode and output.
    if((m_bFirstOutput && m_FrameInfoQueue.size() >= 1) || (!m_bFirstOutput && m_FrameInfoQueue.size()))
    {
        FrameInfo *nOutputFrame = m_FrameInfoQueue.front();
        *pDestBuffer = nOutputFrame->pData;
        *pDestBufferSize = nOutputFrame->nDataSize;
	    *pbIsKeyFrame = nOutputFrame->bIsKeyFrame;
        nOutputFrame->bIsUsed = false;
        m_FrameInfoQueue.pop();
        if(m_bFirstOutput)
            m_bFirstOutput = false;   
    }

#if(DEBUG_INFO)
    cout << "Thread Encode output end"<< endl;
#endif
    pthread_mutex_unlock(&m_OutputMutex);

    return 0;
}

int CNvTegraEncode::InsertKeyFrame()
{
    int hr = 0;
    
    hr = m_nCtx.enc->forceIDR();	    
    
    return hr;
}

int CNvTegraEncode::GetFreeIndex()
{
    /*for(unsigned int i = 0 ; i < m_FrameInfoArray.size() ; i++)
    {
        if(m_FrameInfoArray[i].bIsUsed)
            return i;
    }
    */
    unsigned int nIndex = 0;    

    if(m_nQueueIndex == m_FrameInfoArray.size())
        m_nQueueIndex = 0;
 
    nIndex = m_nQueueIndex;
    m_nQueueIndex++;
    
    return nIndex;

    /*
    FrameInfo nFrameInfo;
    memset(&nFrameInfo, 0, sizeof(FrameInfo));

    nFrameInfo.nMaxDataSize = 100;
    nFrameInfo.pData = new unsigned char[100];

    m_FrameInfoArray.push_back(nFrameInfo);
    
    return m_FrameInfoArray.size() - 1;
    */
}

void CNvTegraEncode::ResizeBuffer(FrameInfo *pFrameInfo, ULONG nSize)
{
    delete [] pFrameInfo->pData;
    pFrameInfo->pData = new unsigned char[nSize];
    pFrameInfo->nMaxDataSize = nSize;
}

void CNvTegraEncode::FrameRateConvert(double dFrameRate, unsigned int *pFrameRateNum, unsigned int *pFrameRateDen)
{
    unsigned int nFrameRate = 0;
    unsigned int nFrameRateNum = 0;
    unsigned int nFrameRateDen = 0;

    nFrameRate = (unsigned int)dFrameRate;
    if(dFrameRate - nFrameRate == 0)
    {
        nFrameRateNum = nFrameRate;
	nFrameRateDen = 1;
    }	
    else if(dFrameRate == 29.97)
    {
        nFrameRateNum = 30000;
        nFrameRateDen = 1001;
    }
    else if(dFrameRate == 59.94)
    {
        nFrameRateNum = 60000;
        nFrameRateDen = 1001;
    }
    else if(dFrameRate == 119.88)
    {
        nFrameRateNum = 120000;
        nFrameRateDen = 1001;
    }
    else if(dFrameRate == 23.976 || dFrameRate == 23.98)
    {
        nFrameRateNum = 24000;
        nFrameRateDen = 1001;
    }
    else
    {
        nFrameRateNum = (unsigned int)((dFrameRate + 0.00005) * 10000);
        nFrameRateDen = 10000;
    }

    *pFrameRateNum = nFrameRateNum;
    *pFrameRateDen = nFrameRateDen;

}

