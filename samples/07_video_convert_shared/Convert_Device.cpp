#include "Convert_Device.h"
#include <cuda_runtime_api.h>
#include "../common/algorithm/cuda/NvCudaProc.h"

using namespace std;

static bool
conv0_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf,
                                   NvBuffer * buffer, NvBuffer * shared_buffer,
                                   void *arg)
{
    context_t *ctx = (context_t *) arg;
    NvBuffer *conv1_buffer;
    struct v4l2_buffer conv1_qbuf;
    struct v4l2_plane planes[MAX_PLANES];

	CNvTegraConvert *pConvertDevice = (CNvTegraConvert *)ctx->m_pConvertDevice;

    if (!v4l2_buf)
    {
        cerr << "Failed to dequeue buffer from conv0 capture plane" << endl;
        pConvertDevice->abort(ctx);
        return false;
    }
    if (ctx->in_buftype ==  BUF_TYPE_NVBL || ctx->out_buftype == BUF_TYPE_NVBL)
    {
        // Get an empty conv1 output plane buffer from conv1_output_plane_buf_queue
        pthread_mutex_lock(&ctx->queue_lock);
        while (ctx->conv1_output_plane_buf_queue->empty() && !ctx->got_error)
        {
            pthread_cond_wait(&ctx->queue_cond, &ctx->queue_lock);
        }

        if (ctx->got_error)
        {
            pthread_mutex_unlock(&ctx->queue_lock);
            return false;
        }
        conv1_buffer = ctx->conv1_output_plane_buf_queue->front();
        ctx->conv1_output_plane_buf_queue->pop();
        pthread_mutex_unlock(&ctx->queue_lock);

        memset(&conv1_qbuf, 0, sizeof(conv1_qbuf));
        memset(&planes, 0, sizeof(planes));

        conv1_qbuf.index = conv1_buffer->index;
        conv1_qbuf.m.planes = planes;

        // A reference to buffer is saved which can be used when
        // buffer is dequeued from conv1 output plane
        if (ctx->conv1->output_plane.qBuffer(conv1_qbuf, buffer)  < 0)
        {
            cerr << "Error queueing buffer on conv1 output plane" << endl;
            pConvertDevice->abort(ctx);
            return false;
        }

        if (v4l2_buf->m.planes[0].bytesused == 0)
        {
            return false;
        }
    }
    else
    {
        if (v4l2_buf->m.planes[0].bytesused == 0)
        {
            return false;
        }

		EGLImageKHR egl_image = NULL;	
		egl_image = NvEGLImageFromFd(ctx->egl_display, buffer->planes[0].fd);
		if(egl_image)
		{
	        mapEGLImage2Float(&egl_image,
	            ctx->out_width,
	            ctx->out_height,
	            COLOR_FORMAT_BGR,
	            (char *)ctx->m_pCudaBuffer);

			if(pConvertDevice->m_fpCallBack)
	        	pConvertDevice->m_fpCallBack(ctx->m_pCudaBuffer, pConvertDevice->m_nCudaBufferSize, pConvertDevice->m_pUserData);
		
	        NvDestroyEGLImage(ctx->egl_display, egl_image);
	        egl_image = NULL;
		}
		else
			printf("Egl image mapping error\n");
#if(0)
	    uint32_t i, j;
	    char *data;

	    for (i = 0; i < buffer->n_planes; i++)
	    {
	        NvBuffer::NvBufferPlane &plane = buffer->planes[i];
	        size_t bytes_to_write =
	            plane.fmt.bytesperpixel * plane.fmt.width;

	        data = (char *) plane.data;
	        for (j = 0; j < plane.fmt.height; j++)
	        {
				fwrite(data, 1, bytes_to_write, ctx->pOutputFile);
	            data += plane.fmt.stride;
	        }
	    }
#endif
        //write_video_frame(ctx->out_file, *buffer);
        if (ctx->conv0->capture_plane.qBuffer(*v4l2_buf, buffer) < 0)
        {
            cerr << "Error queueing buffer on conv0 capture plane" << endl;
            pConvertDevice->abort(ctx);
            return false;
        }
    }
    return true;
}

CNvTegraConvert::CNvTegraConvert() :
	m_nFrameCount(0),
	m_fpCallBack(NULL),
	m_pUserData(NULL),
	m_nCudaBufferSize(0)
{
}

CNvTegraConvert::~CNvTegraConvert()
{
}

int CNvTegraConvert::Create()
{
	memset(&ctx, 0, sizeof(context_t));
	ctx.m_pConvertDevice = (void *)this;
    pthread_mutex_init(&ctx.queue_lock, NULL);
    pthread_cond_init(&ctx.queue_cond, NULL);

#if(0)
	ctx.pOutputFile = fopen(".//OutputRGB.yuv", "wb");	
#endif

	return 0;
}

void CNvTegraConvert::Release()
{
	int ret = 0;

	ret = ConvertFrame(NULL, 0);
    
	if (!ctx.got_error)
    {
        // Wait till all capture plane buffers on conv0 and conv1 are dequeued
        ctx.conv0->waitForIdle(-1);
        if (ctx.conv1)
        {
            ctx.conv1->waitForIdle(-1);
        }
    }

	delete ctx.conv0;

    if (ctx.egl_display)
    {
        if (!eglTerminate(ctx.egl_display))
        {
            printf("Terminate egl display error");
        }
    }

    if(ctx.m_pCudaBuffer)
    {
		cudaFree(ctx.m_pCudaBuffer);
		ctx.m_pCudaBuffer = NULL;
    }

#if(0)
	if(ctx.pOutputFile)
	{
		fclose(ctx.pOutputFile);
		ctx.pOutputFile = NULL;
	}
#endif

}

int CNvTegraConvert::SetFormat(INPUT_COLOR_FORMAT nColorFormat, unsigned int nInputWidth, unsigned int nInputHeight, unsigned int nOutputWidth, unsigned int nOutputHeight, unsigned int nCropLeft, unsigned int nCropTop, unsigned int nCropWidth, unsigned int nCropHeight)
{
	int ret = 0;

	if(!nInputWidth || !nInputHeight || !nOutputWidth || !nOutputHeight)
	{
		printf("nInputWidth:%d nInputHeight:%d nOutputWidth:%d nOutputHeight:%d\n",nInputWidth,nInputHeight,nOutputWidth,nOutputHeight);
		return -1;
	}

	m_nChangeFormat = 0;
	if(nColorFormat == INPUT_COLOR_FORMAT_YV12)
		ctx.in_pixfmt = V4L2_PIX_FMT_YVU420M;
	else if(nColorFormat == INPUT_COLOR_FORMAT_YUY2)
		ctx.in_pixfmt = V4L2_PIX_FMT_YUYV;
    else if(nColorFormat == INPUT_COLOR_FORMAT_NV12)
        ctx.in_pixfmt = V4L2_PIX_FMT_NV12M;
    else if(nColorFormat == INPUT_COLOR_FORMAT_ABGR)
        ctx.in_pixfmt = V4L2_PIX_FMT_ABGR32;
    else if(nColorFormat == INPUT_COLOR_FORMAT_XRGB)
        ctx.in_pixfmt = V4L2_PIX_FMT_XRGB32;
	else if(nColorFormat == INPUT_COLOR_FORMAT_RGB)
	{
		ctx.in_pixfmt = V4L2_PIX_FMT_ABGR32;
		m_nChangeFormat = 1;
	}
	else if(nColorFormat == INPUT_COLOR_FORMAT_BGR)
	{
		ctx.in_pixfmt = V4L2_PIX_FMT_ABGR32;
		m_nChangeFormat = 2;
	}
	else if(nColorFormat == INPUT_COLOR_FORMAT_ARGB)
	{
		ctx.in_pixfmt = V4L2_PIX_FMT_ABGR32;
		m_nChangeFormat = 3;
	}
	else
	{
		printf("Dont support this format:%d\n",nColorFormat);
		return -1;
	}	
	ctx.out_pixfmt = V4L2_PIX_FMT_ABGR32;

	ctx.in_width = nInputWidth;
	ctx.in_height = nInputHeight;
	ctx.out_width = nOutputWidth;
	ctx.out_height = nOutputHeight;
	m_nCudaBufferSize = ctx.out_width * ctx.out_height * sizeof(float) * 3;

    ctx.egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (ctx.egl_display == EGL_NO_DISPLAY)
    {
        printf("Get egl display error\n");
		Release();
        return -1;
    }

    if (!eglInitialize(ctx.egl_display, NULL, NULL))
    {
        printf("Initialize egl display error\n");
		Release();        
		return -1;
    }

    cudaMalloc(&ctx.m_pCudaBuffer, m_nCudaBufferSize);	
    if(!ctx.m_pCudaBuffer)
    {
		//cout << "Allocate cuda buffer error" << endl;
		printf("m_pCudaBuffer error\n");
		Release();
		return -1;
    }

	ctx.conv0 = NvVideoConverter::createVideoConverter("conv0");
    if(!ctx.conv0)
    {
		printf("ctx.conv0 error\n");
		Release();
		return -1;
    }

    if((nCropWidth!=0) && (nCropHeight!=0))
    {
        ret = ctx.conv0->setCropRect(nCropLeft, nCropTop, nCropWidth, nCropHeight);
        if(ret < 0)
        {
            Release();
            cout << "setCropRect = " << ret << endl;
            return -1;
        }  
    }              

    ret = ctx.conv0->setOutputPlaneFormat(ctx.in_pixfmt, ctx.in_width,
                ctx.in_height, V4L2_NV_BUFFER_LAYOUT_PITCH);
    if(ret < 0)
    {
		printf("setOutputPlaneFormat error\n");
		Release();
		return -1;
    }

    ret = ctx.conv0->setCapturePlaneFormat(ctx.out_pixfmt, ctx.out_width,
                ctx.out_height, V4L2_NV_BUFFER_LAYOUT_PITCH);
    if(ret < 0)
    {
		printf("setCapturePlaneFormat error\n");
		Release();
		return -1;
    }

	ret = ctx.conv0->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    if(ret < 0)
    {
		printf("output_plane.setupPlane error\n");
		Release();
		return -1;
    }
    
	ret = ctx.conv0->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false);
    if(ret < 0)
    {
		printf("capture_plane.setupPlane error\n");
		Release();
		return -1;
    }

    ret = ctx.conv0->output_plane.setStreamStatus(true);
    if(ret < 0)
    {
		printf("output_plane.setStreamStatus error\n");
		Release();
		return -1;
    }

    ret = ctx.conv0->capture_plane.setStreamStatus(true);
    if(ret < 0)
    {
		printf("capture_plane.setStreamStatus error\n");
		Release();
		return -1;
    }

    ctx.conv0->capture_plane.setDQThreadCallback(conv0_capture_dqbuf_thread_callback);
	ctx.conv0->capture_plane.startDQThread(&ctx);
	
    for (uint32_t i = 0; i < ctx.conv0->capture_plane.getNumBuffers(); i++)
    {
        struct v4l2_buffer v4l2_buf;
        struct v4l2_plane planes[MAX_PLANES];

        memset(&v4l2_buf, 0, sizeof(v4l2_buf));
        memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;

        ret = ctx.conv0->capture_plane.qBuffer(v4l2_buf, NULL);
        if (ret < 0)
        {
            cout << "Error while queueing buffer at conv0 capture plane" << endl;
            abort(&ctx);
			Release();
			return -1;
        }
    }

	return ret;
}

int CNvTegraConvert::ConvertFrame(uint8_t *pSrcBuffer, unsigned int nSrcBufferSize)
{
	int ret = 0;
	int x = 0;
	uint8_t *pConverterBuffer;
	unsigned int iConvertSize = ctx.in_width*ctx.in_height*4;

	if(m_nChangeFormat)
	{
		pConverterBuffer = (uint8_t*)malloc(iConvertSize);
		if(pSrcBuffer)
		{
			uint8_t *pInputBuffer;
			uint8_t *pOutputBuffer;
			cudaMalloc((void**) &pInputBuffer, nSrcBufferSize);
			cudaMalloc((void**) &pOutputBuffer, iConvertSize);	// ABGR buffer
			cudaMemcpy(pInputBuffer,pSrcBuffer, nSrcBufferSize, cudaMemcpyHostToDevice);			
			if(m_nChangeFormat == 1) 		// RGB -> ABGR
			{
				colorConvert((void*)pInputBuffer,
					ctx.in_width,
					ctx.in_height,
					COLOR_FORMAT_RGBTOABGR,
					(char *)pOutputBuffer);
			}
			else if(m_nChangeFormat == 2) // BGR -> ABGR
			{
				colorConvert((void*)pInputBuffer,
					ctx.in_width,
					ctx.in_height,
					COLOR_FORMAT_BGRTOABGR,
					(char *)pOutputBuffer);
			}
			else 						// ARGB -> ABGR
			{
				colorConvert((void*)pInputBuffer,
					ctx.in_width,
					ctx.in_height,
					COLOR_FORMAT_ARGBTOABGR,
					(char *)pOutputBuffer);
			}
			
			cudaMemcpy(pConverterBuffer,pOutputBuffer,iConvertSize,cudaMemcpyDeviceToHost);
			cudaFree(pInputBuffer);
			cudaFree(pOutputBuffer);			
		}
	}

	m_nFrameCount++;
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];
    NvBuffer *buffer;
    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    if(m_nFrameCount <= ctx.conv0->output_plane.getNumBuffers())
        v4l2_buf.index = m_nFrameCount - 1;

    v4l2_buf.m.planes = planes;

	if(m_nFrameCount <= ctx.conv0->output_plane.getNumBuffers())
	{
		buffer = ctx.conv0->output_plane.getNthBuffer(m_nFrameCount - 1);
	}
	else if (ctx.conv0->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1) < 0)
    {
        abort(&ctx);
        return -1;	
    }

    if(pSrcBuffer)
    {
		uint8_t *pSrcData;
    	NvBuffer::NvBufferPlane *plane = NULL;
		if(m_nChangeFormat)
			pSrcData = pConverterBuffer;
		else
    		pSrcData = pSrcBuffer;
    	char *data = NULL;

		for(unsigned int i = 0 ; i < buffer->n_planes ; i++)
		{
			plane = &(buffer->planes[i]);
			data = (char *)plane->data;
			plane->bytesused = 0;

			unsigned int readwidth = plane->fmt.bytesperpixel * plane->fmt.width;
			for (unsigned int j = 0 ; j < plane->fmt.height ; j++)
			{
				memcpy(data, pSrcData, readwidth);
				pSrcData += readwidth;
				data += plane->fmt.stride;	
			}
			plane->bytesused = plane->fmt.stride * plane->fmt.height;
		}
	}

	if(m_nChangeFormat)
		buffer->planes[0].bytesused = iConvertSize;
	else
		buffer->planes[0].bytesused = nSrcBufferSize;
    v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
	ret = ctx.conv0->output_plane.qBuffer(v4l2_buf, NULL);
	if (ret < 0)
	{
		cerr << "Error while queueing buffer at conv0 output plane" << endl;
		abort(&ctx);
		return -1;
	}

	if(m_nChangeFormat)
		free(pConverterBuffer);
	return ret;
}

void CNvTegraConvert::SetCallBack(NVTegraConvert_Callback cb, void *pUserData)
{
	m_fpCallBack = cb;
	m_pUserData = pUserData;
}	

void CNvTegraConvert::abort(context_t * ctx)
{
    ctx->got_error = true;
    ctx->conv0->abort();
    if (ctx->conv1)
    {
        ctx->conv1->abort();
    }
    pthread_cond_broadcast(&ctx->queue_cond);
}


