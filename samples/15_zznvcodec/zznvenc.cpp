#include "zznvcodec.h"
#include "NvVideoEncoder.h"
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

ZZ_INIT_LOG("zznvenc");

struct zznvcodec_encoder_t {
	enum {
		STATE_READY,
		STATE_STARTED,
	} mState;

	NvVideoEncoder* mEncoder;

	int mWidth;
	int mHeight;
	zznvcodec_pixel_format_t mFormat;
	zznvcodec_encoder_on_video_packet_t mOnVideoPacket;
	intptr_t mOnVideoPacket_User;

	zznvcodec_pixel_format_t mEncoderPixFormat;
	int mBitRate;
	int mProfile;
	v4l2_mpeg_video_h264_level mLevel;
	v4l2_mpeg_video_bitrate_mode mRateControl;
	int mIDRInterval;
	int mIFrameInterval;
	int mFrameRateNum;
	int mFrameRateDeno;

	zznvcodec_video_frame_t mYUY2VideoFrame; // NPP memory
	zznvcodec_video_frame_t mYV12VideoFrame; // NPP memory

	int mMaxPreloadBuffers;
	int mPreloadBuffersIndex;
	int mOutputPlaneFDs[32];

	explicit zznvcodec_encoder_t() {
		mState = STATE_READY;

		mEncoder = NULL;

		mWidth = 0;
		mHeight = 0;
		mFormat = ZZNVCODEC_PIXEL_FORMAT_UNKNOWN;
		mOnVideoPacket = NULL;
		mOnVideoPacket_User = 0;

		mEncoderPixFormat = ZZNVCODEC_PIXEL_FORMAT_UNKNOWN;
		mBitRate = 8 * 1000000;
		mProfile = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
		mLevel = V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
		mRateControl = V4L2_MPEG_VIDEO_BITRATE_MODE_CBR;
		mIDRInterval = 60;
		mIFrameInterval = 60;
		mFrameRateNum = 60;
		mFrameRateDeno = 1;
		memset(&mYUY2VideoFrame, 0, sizeof(mYUY2VideoFrame));
		memset(&mYV12VideoFrame, 0, sizeof(mYV12VideoFrame));

		mMaxPreloadBuffers = 10;
		mPreloadBuffersIndex = 0;
		memset(mOutputPlaneFDs, -1, sizeof(mOutputPlaneFDs));
	}

	~zznvcodec_encoder_t() {
		if(mState != STATE_READY) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
		}
	}

	void SetVideoProperty(int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat) {
		mWidth = nWidth;
		mHeight = nHeight;
		mFormat = nFormat;
	}

	void SetMiscProperty(int nProperty, intptr_t pValue) {
		switch(nProperty) {
		case ZZNVCODEC_PROP_ENCODER_PIX_FMT:
			mEncoderPixFormat = *(zznvcodec_pixel_format_t*)pValue;
			break;

		case ZZNVCODEC_PROP_BITRATE:
			mBitRate = *(int*)pValue;
			break;

		case ZZNVCODEC_PROP_PROFILE:
			mProfile = *(int*)pValue;
			break;

		case ZZNVCODEC_PROP_LEVEL:
			mLevel = (v4l2_mpeg_video_h264_level)*(int*)pValue;
			break;

		case ZZNVCODEC_PROP_RATECONTROL:
			mRateControl = (v4l2_mpeg_video_bitrate_mode)*(int*)pValue;
			break;

		case ZZNVCODEC_PROP_IDRINTERVAL:
			mIDRInterval = *(int*)pValue;
			break;

		case ZZNVCODEC_PROP_IFRAMEINTERVAL:
			mIFrameInterval = *(int*)pValue;
			break;

		case ZZNVCODEC_PROP_FRAMERATE:
			mFrameRateNum = ((int*)pValue)[0];
			mFrameRateDeno = ((int*)pValue)[1];
			break;

		default:
			LOGE("%s(%d): unexpected value, nProperty = %d", __FUNCTION__, __LINE__, nProperty);
			break;
		}
	}

	void RegisterCallbacks(zznvcodec_encoder_on_video_packet_t pCB, intptr_t pUser) {
		mOnVideoPacket = pCB;
		mOnVideoPacket_User = pUser;
	}

	int SetupOutputDMABuf(uint32_t num_buffers) {
		int ret=0;
		NvBufferCreateParams cParams;
		int fd;
		ret = mEncoder->output_plane.reqbufs(V4L2_MEMORY_DMABUF, num_buffers);
		if(ret) {
			LOGE("%s(%d): reqbufs failed for output plane V4L2_MEMORY_DMABUF", __FUNCTION__, __LINE__);
			return ret;
		}

		for (uint32_t i = 0; i < mEncoder->output_plane.getNumBuffers(); i++)
		{
			cParams.width = mWidth;
			cParams.height = mHeight;
			cParams.layout = NvBufferLayout_Pitch;
			cParams.colorFormat = NvBufferColorFormat_YUV420;
			cParams.nvbuf_tag = NvBufferTag_VIDEO_ENC;
			cParams.payloadType = NvBufferPayload_SurfArray;
			ret = NvBufferCreateEx(&fd, &cParams);
			if(ret < 0) {
				LOGE("%s(%d): Failed to create NvBuffer", __FUNCTION__, __LINE__);
				return ret;
			}
			mOutputPlaneFDs[i] = fd;
		}
		return ret;
	}

	static bool _EncoderCapturePlaneDQCallback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer, NvBuffer * shared_buffer, void *arg) {
		zznvcodec_encoder_t* pThis = (zznvcodec_encoder_t*)arg;

		return pThis->EncoderCapturePlaneDQCallback(v4l2_buf, buffer, shared_buffer);
	}

	bool EncoderCapturePlaneDQCallback(struct v4l2_buffer *v4l2_buf, NvBuffer * buffer, NvBuffer * shared_buffer) {
		if (v4l2_buf == NULL) {
			LOGE("%s(%d): Error while dequeing buffer from output plane", __FUNCTION__, __LINE__);
			return false;
		}

		if (buffer->planes[0].bytesused == 0) {
			LOGD("%s(%d): Got 0 size buffer in capture", __FUNCTION__, __LINE__);
			return false;
		}

		int flags = 0;
		v4l2_ctrl_videoenc_outputbuf_metadata enc_metadata;
		if (mEncoder->getMetadata(v4l2_buf->index, enc_metadata) == 0) {
			if(enc_metadata.KeyFrame) {
				flags = 1;
			}
		}

		if(mOnVideoPacket) {
			int64_t pts = v4l2_buf->timestamp.tv_sec * 1000000LL + v4l2_buf->timestamp.tv_usec;
			mOnVideoPacket((uint8_t*)buffer->planes[0].data, buffer->planes[0].bytesused, flags, pts, mOnVideoPacket_User);
		}

		if (mEncoder->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
			LOGE("%s(%d): Error while Qing buffer at capture plane", __FUNCTION__, __LINE__);
			return false;
		}

		return true;
	}

	int Start() {
		int ret;
		NppStatus status;

		if(mState != STATE_READY) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
			return 0;
		}

		LOGD("Start....");

		mEncoder = NvVideoEncoder::createVideoEncoder("enc0");
		if(! mEncoder) {
			LOGE("%s(%d): NvVideoEncoder::createVideoEncoder() failed", __FUNCTION__, __LINE__);
		}

		ret = mEncoder->setCapturePlaneFormat(mEncoderPixFormat, mWidth, mHeight, 2 * 1024 * 1024);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setCapturePlaneFormat() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		if(mFormat == ZZNVCODEC_PIXEL_FORMAT_YUYV422) {
			LOGD("prepare buffer for YUY2 to YV12 conversion");
			{
				mYUY2VideoFrame.num_planes = 1;
				zznvcodec_video_plane_t& plane0 = mYUY2VideoFrame.planes[0];
				plane0.width = mWidth;
				plane0.height = mHeight;
				plane0.ptr = nppiMalloc_8u_C2(plane0.width, plane0.height, &plane0.stride);
			}

			{
				mYV12VideoFrame.num_planes = 3;
				zznvcodec_video_plane_t& plane0 = mYV12VideoFrame.planes[0];
				plane0.width = mWidth;
				plane0.height = mHeight;
				plane0.ptr = nppiMalloc_8u_C1(plane0.width, plane0.height, &plane0.stride);

				zznvcodec_video_plane_t& plane1 = mYV12VideoFrame.planes[1];
				plane1.width = mWidth / 2;
				plane1.height  = mHeight / 2;
				plane1.ptr = nppiMalloc_8u_C1(plane1.width, plane1.height, &plane1.stride);

				zznvcodec_video_plane_t& plane2 = mYV12VideoFrame.planes[2];
				plane2.width = mWidth / 2;
				plane2.height  = mHeight / 2;
				plane2.ptr = nppiMalloc_8u_C1(plane2.width, plane2.height, &plane2.stride);
			}
		}
		ret = mEncoder->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, mWidth, mHeight);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setOutputPlaneFormat() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setBitrate(mBitRate);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setBitrate() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setProfile(mProfile);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setProfile() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setLevel(mLevel);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setLevel() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setRateControlMode(mRateControl);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setRateControlMode() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setIDRInterval(mIDRInterval);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setIDRInterval() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setIFrameInterval(mIFrameInterval);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setIFrameInterval() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setFrameRate(mFrameRateNum, mFrameRateDeno);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setFrameRate() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->setInsertSpsPpsAtIdrEnabled(true);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setInsertSpsPpsAtIdrEnabled() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = SetupOutputDMABuf(mMaxPreloadBuffers);
		if(ret != 0) {
			LOGE("%s(%d): SetupOutputDMABuf() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP, mMaxPreloadBuffers, true, false);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->capture_plane.setupPlane(V4L2_MEMORY_MMAP) failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->subscribeEvent(V4L2_EVENT_EOS, 0, 0);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->subscribeEvent(V4L2_EVENT_EOS) failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		// output plane STREAMON
		ret = mEncoder->output_plane.setStreamStatus(true);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->output_plane.setStreamStatus() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		// capture plane STREAMON
		ret = mEncoder->capture_plane.setStreamStatus(true);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->capture_plane.setStreamStatus() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		mEncoder->capture_plane.setDQThreadCallback(_EncoderCapturePlaneDQCallback);
		mEncoder->capture_plane.startDQThread(this);

		ret = 1;
		// Enqueue all the empty capture plane buffers
		for (uint32_t i = 0; i < mEncoder->capture_plane.getNumBuffers(); i++)
		{
			struct v4l2_buffer v4l2_buf;
			struct v4l2_plane planes[MAX_PLANES];

			memset(&v4l2_buf, 0, sizeof(v4l2_buf));
			memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

			v4l2_buf.index = i;
			v4l2_buf.m.planes = planes;

			ret = mEncoder->capture_plane.qBuffer(v4l2_buf, NULL);
			if (ret < 0) {
				LOGE("%s(%d): Error while queueing buffer at capture plane", __FUNCTION__, __LINE__);
				ret = 0;
				break;
			}
		}

		mState = STATE_STARTED;

		return ret;
	}

	void Stop() {
		int ret;

		if(mState != STATE_STARTED) {
			LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
			return;
		}

		LOGD("Stop....");

		ret = mEncoder->setEncoderCommand(V4L2_ENC_CMD_STOP, 1);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->setEncoderCommand(V4L2_ENC_CMD_STOP) failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->capture_plane.waitForDQThread(3000);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->capture_plane.waitForDQThread() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

		ret = mEncoder->output_plane.waitForDQThread(3000);
		if(ret != 0) {
			LOGE("%s(%d): mEncoder->output_plane.waitForDQThread() failed, err=%d", __FUNCTION__, __LINE__, ret);
		}

        for (uint32_t i = 0; i < mEncoder->output_plane.getNumBuffers(); i++) {
            ret = mEncoder->output_plane.unmapOutputBuffers(i, mOutputPlaneFDs[i]);
            if (ret < 0) {
                LOGE("%s(%d): Error while unmapping buffer at output plane", __FUNCTION__, __LINE__);
            }
			// LOGD("unmapOutputBuffers(%d)", i);

            ret = NvBufferDestroy(mOutputPlaneFDs[i]);
            if(ret < 0) {
                LOGE("%s(%d): Failed to Destroy NvBuffer", __FUNCTION__, __LINE__);
            }
			// LOGD("NvBufferDestroy(%d)", i);
        }
        memset(mOutputPlaneFDs, 0, sizeof(mOutputPlaneFDs));

        delete mEncoder;
		// LOGD("delete mEncoder");
        mEncoder = NULL;

		mPreloadBuffersIndex = 0;

		for(int i = 0;i < mYUY2VideoFrame.num_planes;++i) {
			nppiFree(mYUY2VideoFrame.planes[i].ptr);
		}
		memset(&mYUY2VideoFrame, 0, sizeof(mYUY2VideoFrame));

		for(int i = 0;i < mYV12VideoFrame.num_planes;++i) {
			nppiFree(mYV12VideoFrame.planes[i].ptr);
		}
		memset(&mYV12VideoFrame, 0, sizeof(mYV12VideoFrame));

        mState = STATE_READY;
	}

	void SetVideoUncompressionBuffer(zznvcodec_video_frame_t* pFrame, int64_t nTimestamp) {
		int ret;
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		NvBuffer *buffer;
		NppStatus status;
		cudaError_t cudaError;

		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

		v4l2_buf.m.planes = planes;

		if(mPreloadBuffersIndex == mEncoder->output_plane.getNumBuffers()) {
			// reused
			ret = mEncoder->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 10);
			if(ret < 0) {
				LOGE("%s(%d): Error DQing buffer at output plane", __FUNCTION__, __LINE__);
			}
		} else {
			// preload
			buffer = mEncoder->output_plane.getNthBuffer(mPreloadBuffersIndex);
			v4l2_buf.index = mPreloadBuffersIndex;
			v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
			v4l2_buf.memory = V4L2_MEMORY_DMABUF;
			ret = mEncoder->output_plane.mapOutputBuffers(v4l2_buf, mOutputPlaneFDs[mPreloadBuffersIndex]);
			if (ret < 0) {
				LOGE("%s(%d): Error while mapping buffer at output plane", __FUNCTION__, __LINE__);
			}

			mPreloadBuffersIndex++;
		}

		if(mFormat == ZZNVCODEC_PIXEL_FORMAT_YUYV422) {
			if(pFrame->num_planes != 1) {
				LOGE("%s(%d): unexpected pFrame->num_planes = %d", __FUNCTION__, __LINE__, pFrame->num_planes);
				return;
			}

			zznvcodec_video_plane_t& srcPlane = pFrame->planes[0];
			zznvcodec_video_plane_t& dstPlaneYUY2 = mYUY2VideoFrame.planes[0];
			cudaError = cudaMemcpy2D(dstPlaneYUY2.ptr, dstPlaneYUY2.stride, 
				srcPlane.ptr, srcPlane.stride, srcPlane.width * 2, srcPlane.height, cudaMemcpyHostToDevice);
			if(cudaError != cudaSuccess) {
				LOGE("%s(%d): cudaMemcpy2D failed, cudaError = %d", __FUNCTION__, __LINE__, cudaError);
			}

			Npp8u* pYV12[3] = { mYV12VideoFrame.planes[0].ptr, mYV12VideoFrame.planes[1].ptr, mYV12VideoFrame.planes[2].ptr };
			int nYV12Step[3] = { mYV12VideoFrame.planes[0].stride, mYV12VideoFrame.planes[1].stride, mYV12VideoFrame.planes[2].stride };
			NppiSize oSizeROI = { dstPlaneYUY2.width, dstPlaneYUY2.height };
			status = nppiYCbCr422ToYCbCr420_8u_C2P3R(dstPlaneYUY2.ptr, dstPlaneYUY2.stride, pYV12, nYV12Step, oSizeROI);
			if(status != 0) {
				LOGE("%s(%d): nppiYCbCr422ToYCbCr420_8u_C2P3R failed, status = %d", __FUNCTION__, __LINE__, status);
			}

			for(int i = 0;i < 3;++i) {
				zznvcodec_video_plane_t& srcPlane = mYV12VideoFrame.planes[i];
				NvBuffer::NvBufferPlane &dstPlane = buffer->planes[i];

				cudaError = cudaMemcpy2D(dstPlane.data, dstPlane.fmt.stride, srcPlane.ptr, srcPlane.stride, 
					srcPlane.width, srcPlane.height, cudaMemcpyDeviceToHost);
				if(cudaError != cudaSuccess) {
					LOGE("%s(%d): cudaMemcpy2D failed, cudaError = %d", __FUNCTION__, __LINE__, cudaError);
				}

				dstPlane.bytesused = dstPlane.fmt.stride * dstPlane.fmt.height;
			}

		} else {
			if(pFrame->num_planes != 3) {
				LOGE("%s(%d): unexpected pFrame->num_planes = %d", __FUNCTION__, __LINE__, pFrame->num_planes);
				return;
			}

			for(int i = 0;i < 3;++i) {
				zznvcodec_video_plane_t& srcPlane = pFrame->planes[i];
				NvBuffer::NvBufferPlane &dstPlane = buffer->planes[i];

				cudaError = cudaMemcpy2D(dstPlane.data, dstPlane.fmt.stride, srcPlane.ptr, srcPlane.stride, 
					srcPlane.width, srcPlane.height, cudaMemcpyHostToHost);
				if(cudaError != cudaSuccess) {
					LOGE("%s(%d): cudaMemcpy2D failed, cudaError = %d", cudaError);
				}

				dstPlane.bytesused = dstPlane.fmt.stride * dstPlane.fmt.height;
			}
		}

		for (uint32_t j = 0 ; j < buffer->n_planes ; j++) {
			ret = NvBufferMemSyncForDevice (buffer->planes[j].fd, j, (void **)&buffer->planes[j].data);
			if (ret < 0) {
				LOGE("%s(%d): Error while NvBufferMemSyncForDevice at output plane for V4L2_MEMORY_DMABUF", __FUNCTION__, __LINE__);
				break;
			}

			v4l2_buf.m.planes[j].bytesused = buffer->planes[j].bytesused;
		}

		v4l2_buf.flags |= V4L2_BUF_FLAG_TIMESTAMP_COPY;
		v4l2_buf.timestamp.tv_sec = (int)(nTimestamp / 1000000);
		v4l2_buf.timestamp.tv_usec = (int)(nTimestamp % 1000000);

		ret = mEncoder->output_plane.qBuffer(v4l2_buf, NULL);
		if (ret < 0) {
			LOGE("%s(%d): Error while queueing buffer at output plane", __FUNCTION__, __LINE__);
		}
	}
};

zznvcodec_encoder_t* zznvcodec_encoder_new() {
	return new zznvcodec_encoder_t();
}

void zznvcodec_encoder_delete(zznvcodec_encoder_t* pThis) {
	delete pThis;
}

void zznvcodec_encoder_set_video_property(zznvcodec_encoder_t* pThis, int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat) {
	pThis->SetVideoProperty(nWidth, nHeight, nFormat);
}

void zznvcodec_encoder_set_misc_property(zznvcodec_encoder_t* pThis, int nProperty, intptr_t pValue) {
	pThis->SetMiscProperty(nProperty, pValue);
}

void zznvcodec_encoder_register_callbacks(zznvcodec_encoder_t* pThis, zznvcodec_encoder_on_video_packet_t pCB, intptr_t pUser) {
	pThis->RegisterCallbacks(pCB, pUser);
}

int zznvcodec_encoder_start(zznvcodec_encoder_t* pThis) {
	return pThis->Start();
}

void zznvcodec_encoder_stop(zznvcodec_encoder_t* pThis) {
	return pThis->Stop();
}

void zznvcodec_encoder_set_video_uncompression_buffer(zznvcodec_encoder_t* pThis, zznvcodec_video_frame_t* pFrame, int64_t nTimestamp) {
	return pThis->SetVideoUncompressionBuffer(pFrame, nTimestamp);
}