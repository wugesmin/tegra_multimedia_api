#include "zznvcodec.h"
#include "NvVideoDecoder.h"
#include "ZzLog.h"

#include "NvUtils.h"
#include <errno.h>
#include <fstream>
#include <iostream>
#include <linux/videodev2.h>
#include <malloc.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <nvbuf_utils.h>
#include <npp.h>
#include <atomic>

#define CHUNK_SIZE 4000000
#define MAX_BUFFERS 32
#define MAX_VIDEO_BUFFERS 4

ZZ_INIT_LOG("zznvdec");

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		!buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		(buffer_ptr[2] == 1))

static int MapDMABuf(int dmabuf_fd, unsigned int planes, void** ppsrc_data)
{
	if (dmabuf_fd <= 0)
		return -1;

	int ret = -1;

	for(unsigned int i = 0;i < planes;++i) {
		ret = NvBufferMemMap(dmabuf_fd, i, NvBufferMem_Read_Write, &ppsrc_data[i]);
		if (ret == 0)
		{
			NvBufferMemSyncForCpu(dmabuf_fd, i, &ppsrc_data[i]);
		}
		else
		{
			LOGE("%s(%d): NvBufferMap failed ret=%d\n", __FUNCTION__, __LINE__, ret);
			return -1;
		}
	}

	return 0;
}

static void UnmapDMABuf(int dmabuf_fd, unsigned int planes, void** ppsrc_data) {
	if (dmabuf_fd <= 0)
		return;

	for(unsigned int i = 0;i < planes;++i) {
		NvBufferMemUnMap(dmabuf_fd, i, &ppsrc_data[i]);
	}
}

static int NextId() {
	static std::atomic<int> n(0);

	return n.fetch_add(1);
}

struct zznvcodec_decoder_t {
	enum {
		STATE_READY,
		STATE_STARTED,
	} mState;

	int mId;

	NvVideoDecoder* mDecoder;
	pthread_t mDecoderThread;

	int mDMABufFDs[MAX_BUFFERS];
	int mNumCapBuffers;
	int mVideoDMAFDs[MAX_VIDEO_BUFFERS];
	zznvcodec_video_frame_t mVideoFrames[MAX_VIDEO_BUFFERS];
	int mCurVideoDMAFDIndex;
	int mFormatWidth;
	int mFormatHeight;
	volatile int mGotEOS;
	int mGotError;

	int mWidth;
	int mHeight;
	zznvcodec_pixel_format_t mFormat;
	zznvcodec_decoder_on_video_frame_t mOnVideoFrame;
	intptr_t mOnVideoFrame_User;
	int mMaxPreloadBuffers;
	int mPreloadBuffersIndex;
	NvBufferColorFormat mBufferColorFormat;
	int mV4L2PixFmt;

	explicit zznvcodec_decoder_t() {
		mState = STATE_READY;

		mId = NextId();
		mDecoder = NULL;
		mDecoderThread = (pthread_t)NULL;

		memset(mDMABufFDs, 0, sizeof(mDMABufFDs));
		mNumCapBuffers = 0;
		memset(mVideoDMAFDs, 0, sizeof(mVideoDMAFDs));
		memset(mVideoFrames, 0, sizeof(mVideoFrames));
		mCurVideoDMAFDIndex = 0;
		mFormatWidth = 0;
		mFormatHeight = 0;
		mGotEOS = 0;
		mGotError = 0;

		mWidth = 0;
		mHeight = 0;
		mFormat = ZZNVCODEC_PIXEL_FORMAT_UNKNOWN;
		mOnVideoFrame = NULL;
		mOnVideoFrame_User = 0;
		mMaxPreloadBuffers = 2;
		mPreloadBuffersIndex = 0;
		mBufferColorFormat = NvBufferColorFormat_Invalid;
		mV4L2PixFmt = 0;
	}

	~zznvcodec_decoder_t() {
		if(mState != STATE_READY) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
		}
	}

	void SetVideoProperty(int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat) {
		mWidth = nWidth;
		mHeight = nHeight;
		mFormat = nFormat;

		switch(mFormat) {
		case ZZNVCODEC_PIXEL_FORMAT_NV12:
			mBufferColorFormat = NvBufferColorFormat_NV12;
			break;

		case ZZNVCODEC_PIXEL_FORMAT_YUV420P:
			mBufferColorFormat = NvBufferColorFormat_YUV420;
			break;

		default:
			LOGE("%s(%d): unexpected value, mFormat=%d", __FUNCTION__, __LINE__, mFormat);
			break;
		}
	}

	void SetMiscProperty(int nProperty, intptr_t pValue) {
		switch(nProperty) {
		case ZZNVCODEC_PROP_ENCODER_PIX_FMT: {
			zznvcodec_pixel_format_t* p = (zznvcodec_pixel_format_t*)pValue;
			switch(*p) {
			case ZZNVCODEC_PIXEL_FORMAT_H264:
				mV4L2PixFmt = V4L2_PIX_FMT_H264;
				break;

			case ZZNVCODEC_PIXEL_FORMAT_H265:
				mV4L2PixFmt = V4L2_PIX_FMT_HEVC;
				break;

			default:
				LOGE("%s(%d): unexpected value, *p = %d", __FUNCTION__, __LINE__, *p);
				break;
			}
		}
			break;

		default:
			LOGE("%s(%d): unexpected value, nProperty = %d", __FUNCTION__, __LINE__, nProperty);
		}
	}

	void RegisterCallbacks(zznvcodec_decoder_on_video_frame_t pCB, intptr_t pUser) {
		mOnVideoFrame = pCB;
		mOnVideoFrame_User = pUser;
	}

	int Start() {
		int ret;

		if(mState != STATE_READY) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
			return 0;
		}

		LOGD("[%d] Start decoder...", mId);

		mDecoder = NvVideoDecoder::createVideoDecoder("dec0");
		if(! mDecoder) {
			LOGE("%s(%d): NvVideoDecoder::createVideoDecoder failed", __FUNCTION__, __LINE__);
		}

		ret = mDecoder->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
		if(ret) {
			LOGE("%s(%d): subscribeEvent failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mDecoder->setOutputPlaneFormat(mV4L2PixFmt, CHUNK_SIZE);
		if(ret) {
			LOGE("%s(%d): setOutputPlaneFormat failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mDecoder->setFrameInputMode(0); // 0 --> NALu-based, 1 --> Chunk-based
		if(ret) {
			LOGE("%s(%d): setFrameInputMode failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mDecoder->output_plane.setupPlane(V4L2_MEMORY_MMAP, mMaxPreloadBuffers, true, false);
		if(ret) {
			LOGE("%s(%d): setupPlane failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mDecoder->output_plane.setStreamStatus(true);
		if(ret) {
			LOGE("%s(%d): setStreamStatus failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = pthread_create(&mDecoderThread, NULL, _DecodeMain, this);
		if(ret) {
			ret = errno;
			LOGE("%s(%d): pthread_create failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		mState = STATE_STARTED;

		LOGD("[%d] Start decoder... DONE", mId);

		return 1;
	}

	void Stop() {
		int ret;

		if(mState != STATE_STARTED) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
			return;
		}

		LOGD("Stop decoder...");

		mGotEOS = 1;
		EnqueuePacket(NULL, 0, 0);
		pthread_join(mDecoderThread, NULL);

		delete mDecoder;
		mDecoder = NULL;

		for(int i = 0 ; i < mNumCapBuffers ; i++) {
			if(mDMABufFDs[i] != 0) {
				ret = NvBufferDestroy (mDMABufFDs[i]);
			}
		}
		memset(mDMABufFDs, 0, sizeof(mDMABufFDs));
		mNumCapBuffers = 0;
		for(int i = 0 ; i < MAX_VIDEO_BUFFERS ; i++) {
			if(mVideoDMAFDs[i] != 0) {
				if(mVideoFrames[i].num_planes != 0) {
					void* pPlanes[ZZNVCODEC_MAX_PLANES] = {
						mVideoFrames[i].planes[0].ptr,
						mVideoFrames[i].planes[1].ptr,
						mVideoFrames[i].planes[2].ptr };
					UnmapDMABuf(mVideoDMAFDs[i], mVideoFrames[i].num_planes, pPlanes);
				}
				ret = NvBufferDestroy (mVideoDMAFDs[i]);
			}
		}
		memset(mVideoDMAFDs, 0, sizeof(mVideoDMAFDs));
		memset(mVideoFrames, 0, sizeof(mVideoFrames));
		mCurVideoDMAFDIndex = 0;
		mFormatWidth = 0;
		mFormatHeight = 0;
		mGotEOS = 0;
		mGotError = 0;
		mMaxPreloadBuffers = 2;
		mPreloadBuffersIndex = 0;
		mBufferColorFormat = NvBufferColorFormat_Invalid;

		mState = STATE_READY;

		LOGD("Stop decoder... DONE");
	}

	void EnqueuePacket(unsigned char* pBuffer, int nSize, int64_t nTimestamp) {
		int ret;
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		NvBuffer *buffer;

		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, sizeof(planes));

		v4l2_buf.m.planes = planes;

		if(mPreloadBuffersIndex == mDecoder->output_plane.getNumBuffers()) {
			// reused
			ret = mDecoder->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1);
			if(ret < 0) {
				LOGE("%s(%d): [%d] Error DQing buffer at output plane", __FUNCTION__, __LINE__, mId);
			}
		} else {
			// preload
			buffer = mDecoder->output_plane.getNthBuffer(mPreloadBuffersIndex);
			v4l2_buf.index = mPreloadBuffersIndex;
			mPreloadBuffersIndex++;
		}

		if(pBuffer) {
			memcpy((char *) buffer->planes[0].data, (Npp8u*)pBuffer, nSize);
		}
		buffer->planes[0].bytesused = nSize;

		v4l2_buf.m.planes[0].bytesused = buffer->planes[0].bytesused;
		v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
		v4l2_buf.timestamp.tv_sec = (int)(nTimestamp / 1000000);
		v4l2_buf.timestamp.tv_usec = (int)(nTimestamp % 1000000);

#if 0
		LOGD("%s(%d): buffer: index=%d planes[0]:bytesused=%d", __FUNCTION__, __LINE__, buffer->index, buffer->planes[0].bytesused);
		LOGD("%s(%d): [%02X %02X %02X %02X %02X %02X %02X %02X]", __FUNCTION__, __LINE__,
			(int)((uint8_t*)buffer->planes[0].data)[0], (int)((uint8_t*)buffer->planes[0].data)[1],
			(int)((uint8_t*)buffer->planes[0].data)[2], (int)((uint8_t*)buffer->planes[0].data)[3],
			(int)((uint8_t*)buffer->planes[0].data)[4], (int)((uint8_t*)buffer->planes[0].data)[5],
			(int)((uint8_t*)buffer->planes[0].data)[6], (int)((uint8_t*)buffer->planes[0].data)[7]);
#endif

		ret = mDecoder->output_plane.qBuffer(v4l2_buf, NULL);
		if (ret < 0)
		{
			LOGE("%s(%d): Error Qing buffer at output plane", __FUNCTION__, __LINE__);
		}
	}

	void SetVideoCompressionBuffer(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp) {
#if 0
		EnqueuePacket(pBuffer, nSize, nTimestamp);
#else
		// find first NALu
		int start_bytes;
		while(nSize > 4) {
			if(IS_NAL_UNIT_START(pBuffer)) {
				start_bytes = 4;
				break;
			} else if(IS_NAL_UNIT_START1(pBuffer)) {
				start_bytes = 3;
				break;
			}

			pBuffer++;
			nSize--;
		}

		// find rest of NALu
		while(true) {
			unsigned char* next_nalu = pBuffer + start_bytes;
			int next_size = nSize - start_bytes;
			int next_start_bytes;
			while(next_size > 4) {
				if(IS_NAL_UNIT_START(next_nalu)) {
					next_start_bytes = 4;
					break;
				} else if(IS_NAL_UNIT_START1(next_nalu)) {
					next_start_bytes = 3;
					break;
				}

				next_nalu++;
				next_size--;
			}

			if(next_size <= 4) {
				// the last NALu
				EnqueuePacket(pBuffer, nSize, nTimestamp);
				break;
			}

			EnqueuePacket(pBuffer, (int)(next_nalu - pBuffer), nTimestamp);
			pBuffer = next_nalu;
			nSize = next_size;
			start_bytes = next_start_bytes;
		}
#endif
	}

	static void* _DecodeMain(void* arg) {
		zznvcodec_decoder_t* pThis = (zznvcodec_decoder_t*)arg;

		return pThis->DecoderMain();
	}

	void* DecoderMain() {
		int ret, err;
		struct v4l2_event ev;

		LOGD("%s: [%d] begins", __FUNCTION__, mId);

		bool got_resolution = false;
		while(! got_resolution && !mGotError) {
			LOGD("%s(%d): [%d] wait for V4L2_EVENT_RESOLUTION_CHANGE...", __FUNCTION__, __LINE__, mId);
			ret = mDecoder->dqEvent(ev, 50000);
			if (ret == 0)
			{
				switch (ev.type)
				{
					case V4L2_EVENT_RESOLUTION_CHANGE:
						LOGD("%s(%d): [%d] New V4L2_EVENT_RESOLUTION_CHANGE received!!", __FUNCTION__, __LINE__, mId);
						QueryAndSetCapture();
						got_resolution = true;
						break;

					default:
						LOGE("%s(%d): [%d] unexpected value, ev.type=%d", __FUNCTION__, __LINE__, mId, ev.type);
						break;
				}
			} else {
				err = errno;
				LOGE("%s(%d): [%d] failed to received V4L2_EVENT_RESOLUTION_CHANGE, err=%d", __FUNCTION__, __LINE__, mId, err);
				mGotError = 1;
			}
		}

		LOGD("[%d] start decoding... error=%d, isInError=%d, EOS=%d", mId, mGotError, mDecoder->isInError(), mGotEOS);
		while (!(mGotError || mDecoder->isInError() || mGotEOS)) {
			NvBuffer *dec_buffer;
			struct v4l2_buffer v4l2_buf;
			struct v4l2_plane planes[MAX_PLANES];

			memset(&v4l2_buf, 0, sizeof(v4l2_buf));
			memset(planes, 0, sizeof(planes));
			v4l2_buf.m.planes = planes;

			// Dequeue a filled buffer
			if (mDecoder->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0))
			{
				err = errno;
				if (err == EAGAIN)
				{
					if (v4l2_buf.flags & V4L2_BUF_FLAG_LAST)
					{
						LOGD("[%d] Got EoS at capture plane", mId);
						break;
					}

					usleep(1000);
					continue;
				}
				else
				{
					if(! mGotEOS) {
						LOGE("%s(%d): Error while calling dequeue at capture plane, errno=%d", __FUNCTION__, __LINE__, err);
					}
				}
				mGotError = 1;
				break;
			}

			/* Clip & Stitch can be done by adjusting rectangle */
			NvBufferRect src_rect, dest_rect;
			src_rect.top = 0;
			src_rect.left = 0;
			src_rect.width = mFormatWidth;
			src_rect.height = mFormatHeight;
			dest_rect.top = 0;
			dest_rect.left = 0;
			dest_rect.width = mFormatWidth;
			dest_rect.height = mFormatHeight;
			NvBufferTransformParams transform_params;
			memset(&transform_params,0,sizeof(transform_params));
			/* Indicates which of the transform parameters are valid */
			transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
			transform_params.transform_flip = NvBufferTransform_None;
			transform_params.transform_filter = NvBufferTransform_Filter_Smart;
			transform_params.src_rect = src_rect;
			transform_params.dst_rect = dest_rect;

			dec_buffer->planes[0].fd = mDMABufFDs[v4l2_buf.index];

			// ring video frame buffer
			int dst_fd = mVideoDMAFDs[mCurVideoDMAFDIndex];
			zznvcodec_video_frame_t& oVideoFrame = mVideoFrames[mCurVideoDMAFDIndex];
			mCurVideoDMAFDIndex = (mCurVideoDMAFDIndex + 1) % MAX_VIDEO_BUFFERS;

			// Convert Blocklinear to PitchLinear
			ret = NvBufferTransform(dec_buffer->planes[0].fd, dst_fd, &transform_params);
			if (ret == -1)
			{
				LOGE("%s(%d): Transform failed", __FUNCTION__, __LINE__);
				mGotError = 1;
				break;
			}

			NvBufferParams parm;
			ret = NvBufferGetParams(dst_fd, &parm);

#if 0
			LOGD("%s(%d): parm={%d(%d) %d, %dx%d(%d) %dx%d(%d) %dx%d(%d)}\n", __FUNCTION__, __LINE__,
				parm.pixel_format, NvBufferColorFormat_YUV420, parm.num_planes,
				parm.width[0], parm.height[0], parm.pitch[0],
				parm.width[1], parm.height[1], parm.pitch[1],
				parm.width[2], parm.height[2], parm.pitch[2]);
#endif

			if(oVideoFrame.num_planes == 0) {
				void* pPlanes[ZZNVCODEC_MAX_PLANES] = { NULL, NULL, NULL };
				ret = MapDMABuf(dst_fd, parm.num_planes, pPlanes);

				oVideoFrame.num_planes = parm.num_planes;
				for(int i = 0;i < parm.num_planes;++i) {
					oVideoFrame.planes[i].width = parm.width[i];
					oVideoFrame.planes[i].height = parm.height[i];
					oVideoFrame.planes[i].ptr = (uint8_t*)pPlanes[i];
					oVideoFrame.planes[i].stride = parm.pitch[i];
				}
			}

#if 0 // DEBUG
			LOGD("%s(%d): dst_fd=%d planes[]={%p %p %p}\n", __FUNCTION__, __LINE__, dst_fd,
				oVideoFrame.planes[0].ptr, oVideoFrame.planes[1].ptr, oVideoFrame.planes[2].ptr);
#endif

			if(! mGotEOS && mOnVideoFrame) {
				int64_t pts = v4l2_buf.timestamp.tv_sec * 1000000LL + v4l2_buf.timestamp.tv_usec;
				mOnVideoFrame(&oVideoFrame, pts, mOnVideoFrame_User);
			}

			v4l2_buf.m.planes[0].m.fd = mDMABufFDs[v4l2_buf.index];
			if (mDecoder->capture_plane.qBuffer(v4l2_buf, NULL) < 0)
			{
				LOGE("%s(%d): Error while queueing buffer at decoder capture plane", __FUNCTION__, __LINE__);
				mGotError = 1;
				break;
			}
		}

		LOGD("%s: [%d] ends", __FUNCTION__, mId);

		return NULL;
	}

	void QueryAndSetCapture() {
		int ret;
		struct v4l2_format format;
		struct v4l2_crop crop;
		int32_t min_dec_capture_buffers;
		NvBufferCreateParams input_params = {0};
		NvBufferCreateParams cParams = {0};

		ret = mDecoder->capture_plane.getFormat(format);
		ret = mDecoder->capture_plane.getCrop(crop);
		LOGD("Video Resolution: %d x %d (PixFmt=%08X, %dx%d)", crop.c.width, crop.c.height,
			format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);

		mDecoder->capture_plane.deinitPlane();
		for(int index = 0 ; index < mNumCapBuffers ; index++) {
			if(mDMABufFDs[index] != 0) {
				ret = NvBufferDestroy (mDMABufFDs[index]);
			}
		}
		memset(mDMABufFDs, 0, sizeof(mDMABufFDs));
		for(int i = 0 ; i < MAX_VIDEO_BUFFERS ; i++) {
			if(mVideoDMAFDs[i] != 0) {
				if(mVideoFrames[i].num_planes != 0) {
					void* pPlanes[ZZNVCODEC_MAX_PLANES] = {
						mVideoFrames[i].planes[0].ptr,
						mVideoFrames[i].planes[1].ptr,
						mVideoFrames[i].planes[2].ptr };
					UnmapDMABuf(mVideoDMAFDs[i], mVideoFrames[i].num_planes, pPlanes);
				}

				ret = NvBufferDestroy (mVideoDMAFDs[i]);
			}
		}
		memset(mVideoDMAFDs, 0, sizeof(mVideoDMAFDs));
		memset(mVideoFrames, 0, sizeof(mVideoFrames));

		ret = mDecoder->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);
		mFormatHeight = format.fmt.pix_mp.height;
		mFormatWidth = format.fmt.pix_mp.width;

		for(int i = 0;i < MAX_VIDEO_BUFFERS;++i) {
			input_params.payloadType = NvBufferPayload_SurfArray;
			input_params.width = crop.c.width;
			input_params.height = crop.c.height;
			input_params.layout = NvBufferLayout_Pitch;
			input_params.colorFormat = mBufferColorFormat;
			input_params.nvbuf_tag = NvBufferTag_VIDEO_CONVERT;

			ret = NvBufferCreateEx (&mVideoDMAFDs[i], &input_params);
		}

		ret = mDecoder->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
		switch(format.fmt.pix_mp.colorspace) {
			case V4L2_COLORSPACE_SMPTE170M:
				if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
				{
					LOGD("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)");
					cParams.colorFormat = NvBufferColorFormat_NV12;
				}
				else
				{
					LOGD("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)");
					cParams.colorFormat = NvBufferColorFormat_NV12_ER;
				}
				break;
			case V4L2_COLORSPACE_REC709:
				if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
				{
					LOGD("Decoder colorspace ITU-R BT.709 with standard range luma (16-235)");
					cParams.colorFormat = NvBufferColorFormat_NV12_709;
				}
				else
				{
					LOGD("Decoder colorspace ITU-R BT.709 with extended range luma (0-255)");
					cParams.colorFormat = NvBufferColorFormat_NV12_709_ER;
				}
				break;
			case V4L2_COLORSPACE_BT2020:
				{
					LOGD("Decoder colorspace ITU-R BT.2020");
					cParams.colorFormat = NvBufferColorFormat_NV12_2020;
				}
				break;
			default:
				LOGD("supported colorspace details not available, use default");
				if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
				{
					LOGD("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)");
					cParams.colorFormat = NvBufferColorFormat_NV12;
				}
				else
				{
					LOGD("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)");
					cParams.colorFormat = NvBufferColorFormat_NV12_ER;
				}
				break;
		}

		mNumCapBuffers = min_dec_capture_buffers + 1;

		for (int index = 0; index < mNumCapBuffers; index++)
		{
			cParams.width = crop.c.width;
			cParams.height = crop.c.height;
			cParams.layout = NvBufferLayout_BlockLinear;
			cParams.payloadType = NvBufferPayload_SurfArray;
			cParams.nvbuf_tag = NvBufferTag_VIDEO_DEC;
			ret = NvBufferCreateEx(&mDMABufFDs[index], &cParams);
		}
		ret = mDecoder->capture_plane.reqbufs(V4L2_MEMORY_DMABUF, mNumCapBuffers);

		ret = mDecoder->capture_plane.setStreamStatus(true);

		for (uint32_t i = 0; i < mDecoder->capture_plane.getNumBuffers(); i++)
		{
			struct v4l2_buffer v4l2_buf;
			struct v4l2_plane planes[MAX_PLANES];

			memset(&v4l2_buf, 0, sizeof(v4l2_buf));
			memset(planes, 0, sizeof(planes));

			v4l2_buf.index = i;
			v4l2_buf.m.planes = planes;
			v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			v4l2_buf.memory = V4L2_MEMORY_DMABUF;
			v4l2_buf.m.planes[0].m.fd = mDMABufFDs[i];
			ret = mDecoder->capture_plane.qBuffer(v4l2_buf, NULL);
		}
	}
};

zznvcodec_decoder_t* zznvcodec_decoder_new() {
	return new zznvcodec_decoder_t();
}

void zznvcodec_decoder_delete(zznvcodec_decoder_t* pThis) {
	delete pThis;
}

void zznvcodec_decoder_set_video_property(zznvcodec_decoder_t* pThis, int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat) {
	pThis->SetVideoProperty(nWidth, nHeight, nFormat);
}

void zznvcodec_decoder_set_misc_property(zznvcodec_decoder_t* pThis, int nProperty, intptr_t pValue) {
	pThis->SetMiscProperty(nProperty, pValue);
}

void zznvcodec_decoder_register_callbacks(zznvcodec_decoder_t* pThis, zznvcodec_decoder_on_video_frame_t pCB, intptr_t pUser) {
	pThis->RegisterCallbacks(pCB, pUser);
}

int zznvcodec_decoder_start(zznvcodec_decoder_t* pThis) {
	return pThis->Start();
}

void zznvcodec_decoder_stop(zznvcodec_decoder_t* pThis) {
	return pThis->Stop();
}

void zznvcodec_decoder_set_video_compression_buffer(zznvcodec_decoder_t* pThis, unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp) {
	return pThis->SetVideoCompressionBuffer(pBuffer, nSize, nFlags, nTimestamp);
}