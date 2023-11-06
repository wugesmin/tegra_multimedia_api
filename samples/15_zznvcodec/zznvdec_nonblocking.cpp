#include "zznvdec.h"
#include "ZzLog.h"

#include "NvVideoDecoder.h"
#include "NvUtils.h"
#include "NvBufSurface.h"

#include <vector>

ZZ_INIT_LOG("zznvdec_nonblocking");

#define _countof(x) (sizeof(x)/sizeof(x[0]))
#define CHUNK_SIZE 4000000
#define MAX_NALUS 64
#define MAX_VIDEO_BUFFERS 2

#define IS_NAL_UNIT_START(buffer_ptr) \
(!buffer_ptr[0] && !buffer_ptr[1] && \
!buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) \
(!buffer_ptr[0] && !buffer_ptr[1] && \
(buffer_ptr[2] == 1))

namespace __zznvdec_nonblocking__ {
	class VideoDecoder : public NvVideoDecoder {
		typedef VideoDecoder self_t;
		typedef NvVideoDecoder super_t;

	public:
		explicit VideoDecoder();
		virtual ~VideoDecoder();

		int GetFD() const {
			return fd;
		}
	};

	inline VideoDecoder* CreateVideoDecoder() {
#if 0
		return NvVideoDecoder::createVideoDecoder("dec0", O_NONBLOCK);
#else
		VideoDecoder* pDecoder = new VideoDecoder();
		if (pDecoder->isInError()) {
			delete pDecoder;
			return NULL;
		}

		return pDecoder;
#endif
	}
}

struct zznvcodec_decoder_nonblocking : public zznvcodec_decoder_t {
	typedef zznvcodec_decoder_nonblocking self_t;
	typedef zznvcodec_decoder_t super_t;

	explicit zznvcodec_decoder_nonblocking();
	virtual ~zznvcodec_decoder_nonblocking();

	virtual int Start();
	virtual void Stop();
	virtual void SetVideoCompressionBuffer(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp);
	virtual void SetVideoCompressionBuffer2(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp, unsigned char *pDestBuffer, int64_t *nDestBufferSize, int64_t *nDestTimestamp);
	virtual int GetVideoCompressionBuffer(zznvcodec_video_frame_t** ppFrame, int64_t* pTimestamp);

	enum {
		STATE_READY,
		STATE_STARTED,
	} mState;

	__zznvdec_nonblocking__::VideoDecoder* mDecoder;
	NvBufSurfaceColorFormat mBufferColorFormat;
	int mV4L2PixFmt;
	int mPreloadPackets;
	bool mGotResolution;
	struct v4l2_format mDecodedFormat;
	int mNumCapBuffers;
	std::vector<int> mDMABuf_Cap;

	void EnqueueNalu(unsigned char* pBuffer, int nSize, int64_t nTimestamp);
	int QueryAndSetCapture();
};

using namespace __zznvdec_nonblocking__;

namespace __zznvdec_nonblocking__ {
	VideoDecoder::VideoDecoder() : super_t("dec0", O_NONBLOCK) {
		LOGD("%s(%d): this=%p", __FUNCTION__, __LINE__, this);
	}

	VideoDecoder::~VideoDecoder() {
		LOGD("%s(%d): this=%p", __FUNCTION__, __LINE__, this);
	}
}

zznvcodec_decoder_t* NewDecoder_nonblocking() {
	return new zznvcodec_decoder_nonblocking();
}

zznvcodec_decoder_nonblocking::zznvcodec_decoder_nonblocking() {
	mState = STATE_READY;

	mDecoder = NULL;
}

zznvcodec_decoder_nonblocking::~zznvcodec_decoder_nonblocking() {
	if(mState != STATE_READY) {
		LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
	}
}

int zznvcodec_decoder_nonblocking::Start() {
	int ret;

	if(mState != STATE_READY) {
		LOGE("%s(%d): unexpected value, mState=%d", __FUNCTION__, __LINE__, mState);
		return 0;
	}

	int err = 0;

	switch(1) { case 1:
		switch(mFormat) {
		case ZZNVCODEC_PIXEL_FORMAT_NV12:
			mBufferColorFormat = NVBUF_COLOR_FORMAT_NV12;
			break;

		case ZZNVCODEC_PIXEL_FORMAT_NV24:
			mBufferColorFormat = NVBUF_COLOR_FORMAT_NV24;
			break;

		case ZZNVCODEC_PIXEL_FORMAT_YUV420P:
			mBufferColorFormat = NVBUF_COLOR_FORMAT_YUV420;
			break;

		default:
			err = -1;
			LOGE("%s(%d): unexpected value, mFormat=%d", __FUNCTION__, __LINE__, mFormat);
			break;
		}

		if(err) break;

		switch(mCodecType) {
		case ZZNVCODEC_CODEC_TYPE_H264:
			mV4L2PixFmt = V4L2_PIX_FMT_H264;
			break;

		case ZZNVCODEC_CODEC_TYPE_H265:
			mV4L2PixFmt = V4L2_PIX_FMT_H265;
			break;

		case ZZNVCODEC_CODEC_TYPE_AV1:
			mV4L2PixFmt = V4L2_PIX_FMT_AV1;
			break;

		default:
			err = -1;
			LOGE("%s(%d): unexpected value, mCodecType=%d", __FUNCTION__, __LINE__, mCodecType);
			break;
		}

		if(err) break;

		// mDecoder = NvVideoDecoder::createVideoDecoder("dec0", O_NONBLOCK);
		mDecoder = CreateVideoDecoder();
		if(! mDecoder) {
			err = errno;
			LOGE("%s(%d): CreateVideoDecoder failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
		if(err) {
			LOGE("%s(%d): mDecoder->subscribeEvent() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->setOutputPlaneFormat(mV4L2PixFmt, CHUNK_SIZE);
		if(err) {
			LOGE("%s(%d): mDecoder->setOutputPlaneFormat() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->setFrameInputMode(0); // 0 --> NALu-based, 1 --> Chunk-based
		if(err) {
			LOGE("%s(%d): mDecoder->setFrameInputMode() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

	    err = mDecoder->disableDPB();
		if(err) {
			LOGE("%s(%d): mDecoder->disableDPB() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->setMaxPerfMode(1);
		if(err) {
			LOGE("%s(%d): setMaxPerfMode failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->output_plane.setupPlane(V4L2_MEMORY_MMAP, MAX_NALUS, true, false);
		if(err) {
			LOGE("%s(%d): mDecoder->output_plane.setupPlane() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->output_plane.setStreamStatus(true);
		if(err) {
			LOGE("%s(%d): mDecoder->output_plane.setStreamStatus() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

#if 0
		err = mDecoder->SetPollInterrupt();
		if(err) {
			LOGE("%s(%d): mDecoder->SetPollInterrupt() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}
#endif

		mPreloadPackets = 0;
		mGotResolution = false;
		mState = STATE_STARTED;
	}

	if(err) {
		Stop();
	}

	return err;
}

void zznvcodec_decoder_nonblocking::Stop() {
	int err;

	TODO_TAG();

	if(mDecoder) {
		delete mDecoder;
		mDecoder = NULL;
	}

	mState = STATE_READY;
}

void zznvcodec_decoder_nonblocking::SetVideoCompressionBuffer(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp) {
	switch(mV4L2PixFmt) {
	case V4L2_PIX_FMT_AV1:
		EnqueueNalu(pBuffer, nSize, nTimestamp);
		break;

	case V4L2_PIX_FMT_H264:
	case V4L2_PIX_FMT_H265: {
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
				EnqueueNalu(pBuffer, nSize, nTimestamp);
				break;
			}

			EnqueueNalu(pBuffer, (int)(next_nalu - pBuffer), nTimestamp);
			pBuffer = next_nalu;
			nSize = next_size;
			start_bytes = next_start_bytes;
		}
	}
		break;

	default:
		LOGE("%s(%d): unexpected value, mV4L2PixFmt=0x%X", __FUNCTION__, __LINE__, mV4L2PixFmt);
		break;
	}
}

void zznvcodec_decoder_nonblocking::SetVideoCompressionBuffer2(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp, unsigned char *pDestBuffer, int64_t *nDestBufferSize, int64_t *nDestTimestamp) {
	TODO_TAG();
}

int zznvcodec_decoder_nonblocking::GetVideoCompressionBuffer(zznvcodec_video_frame_t** ppFrame, int64_t* pTimestamp) {
	int err = 0;

	switch(1) { case 1:
		if(! mGotResolution) {
		    struct v4l2_event ev;

			err = mDecoder->dqEvent(ev, 10000);
			if (err == 0) {
				if (ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
					err = QueryAndSetCapture();
					if(err) {
						LOGE("%s(%d): QueryAndSetCapture() failed, err=%d", __FUNCTION__, __LINE__, err);
						break;
					}

					mGotResolution = true;
				} else {
					LOGD("mDecoder->dqEvent(), ev.type=%d", ev.type);
					break;
				}
			} else {
				LOGE("%s(%d): mDecoder->dqEvent(() failed, err=%d", __FUNCTION__, __LINE__, err);
				break;
			}
		}

		if(mDecoder->isInError()) {
			LOGE("%s(%d): mDecoder->isInError()", __FUNCTION__, __LINE__);
			err = -1;
			break;
		}

		if(! mDecoder->capture_plane.getStreamStatus()) {
			LOGE("%s(%d): ! mDecoder->capture_plane.getStreamStatus()", __FUNCTION__, __LINE__);
			err = -1;
			break;
		}

#if 0
		v4l2_ctrl_video_device_poll devicepoll;
		memset(&devicepoll, 0, sizeof(v4l2_ctrl_video_device_poll));

TRACE_TAG();
		devicepoll.req_events = POLLIN | POLLERR | POLLPRI;
		err = mDecoder->DevicePoll(&devicepoll);
		if(err < 0) {
			LOGE("%s(%d): ! mDecoder->DevicePoll() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}
TRACE_TAG();
#else
		int fd_vdec = mDecoder->GetFD();
LOGW("%s(%d): fd_vdec=%d", __FUNCTION__, __LINE__, fd_vdec);

		fd_set readfds;
		FD_ZERO(&readfds);

		int fd_max = -1;
		if(fd_vdec > fd_max) fd_max = fd_vdec;
		FD_SET(fd_vdec, &readfds);
		int err = select(fd_max + 1, &readfds, NULL, NULL, NULL);
		if (err < 0) {
			LOGE("%s(%d): select failed! err = %d", __FUNCTION__, __LINE__, err);
			break;
		}

TRACE_TAG();
#endif

		NvBuffer *dec_buffer;
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];

		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, sizeof(planes));
		v4l2_buf.m.planes = planes;

		if (mDecoder->capture_plane.dqBuffer(v4l2_buf, &dec_buffer, NULL, 0)) {
			err = errno;
			if(err == EAGAIN) {
TRACE_TAG();
				break;
			}

			LOGD("%s(%d): mDecoder->capture_plane.dqBuffer(), err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		if(! dec_buffer) {
			LOGE("%s(%d): unexpected value, dec_buffer=null", __FUNCTION__, __LINE__);
			break;
		}

		v4l2_buf.m.planes[0].m.fd = mDMABuf_Cap[v4l2_buf.index];

		int64_t pts = v4l2_buf.timestamp.tv_sec * 1000000LL + v4l2_buf.timestamp.tv_usec;

#if 1 // DEBUG
		LOGD("v4l2_buf.index=%d, pts=%.2f, fd=%d", v4l2_buf.index, pts / 1000.0, v4l2_buf.m.planes[0].m.fd);
#endif

		if (mDecoder->capture_plane.qBuffer(v4l2_buf, NULL) < 0) {
			err = errno;
			LOGE("%s(%d): ! mDecoder->capture_plane.qBuffer(), err=%d", __FUNCTION__, __LINE__, err);
			break;
		}
	}

	return err;
}

void zznvcodec_decoder_nonblocking::EnqueueNalu(unsigned char* pBuffer, int nSize, int64_t nTimestamp) {
	int err;

	switch(1) { case 1:
		struct v4l2_buffer v4l2_buf;
		struct v4l2_plane planes[MAX_PLANES];
		NvBuffer *buffer;

		memset(&v4l2_buf, 0, sizeof(v4l2_buf));
		memset(planes, 0, sizeof(planes));

		v4l2_buf.m.planes = planes;

		if(mPreloadPackets == mDecoder->output_plane.getNumBuffers()) {
			// reused
			err = mDecoder->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, 0);
			if(err < 0) {
				LOGE("%s(%d): mDecoder->output_plane.dqBuffer() failed, err=%d", __FUNCTION__, __LINE__, err);
				break;
			}
		} else {
			// preload
			buffer = mDecoder->output_plane.getNthBuffer(mPreloadPackets);
			v4l2_buf.index = mPreloadPackets;
			mPreloadPackets++;
		}

		if(pBuffer) {
			memcpy(buffer->planes[0].data, pBuffer, nSize);
		} else {
			v4l2_buf.flags |= V4L2_BUF_FLAG_LAST;
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

		err = mDecoder->output_plane.qBuffer(v4l2_buf, NULL);
		if(err < 0) {
			LOGE("%s(%d): mDecoder->output_plane.qBuffer() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}
	}
}

int zznvcodec_decoder_nonblocking::QueryAndSetCapture() {
	int err = 0;
	struct v4l2_format format;
	struct v4l2_crop crop;
	uint32_t sar_width;
	uint32_t sar_height;
	int32_t min_dec_capture_buffers;
	NvBufSurf::NvCommonAllocateParams params;
	NvBufSurf::NvCommonAllocateParams capParams;

	switch(1) { case 1:
		err = mDecoder->capture_plane.getFormat(format);
		if(err) {
			LOGE("%s(%d): mDecoder->capture_plane.getFormat(), err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->capture_plane.getCrop(crop);
		if(err) {
			LOGE("%s(%d): mDecoder->capture_plane.getCrop(), err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		LOGD("Video Resolution: %d x %d (PixFmt=%08X, %dx%d)", crop.c.width, crop.c.height,
			format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);

		err = mDecoder->getSAR(sar_width, sar_height);
		if(err) {
			LOGE("%s(%d): mDecoder->getSAR(), err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		LOGD("Video SAR: %d x %d", sar_width, sar_height);

		/* deinitPlane unmaps the buffers and calls REQBUFS with count 0 */
		mDecoder->capture_plane.deinitPlane();

		err = mDecoder->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat, format.fmt.pix_mp.width, format.fmt.pix_mp.height);
		if(err < 0) {
			LOGE("%s(%d): mDecoder->setCapturePlaneFormat failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		err = mDecoder->getMinimumCapturePlaneBuffers(min_dec_capture_buffers);
		if(err < 0) {
			LOGE("%s(%d): mDecoder->getMinimumCapturePlaneBuffers failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		min_dec_capture_buffers += 1;
#if 0 // DEBUG
		LOGD("!!!!! min_dec_capture_buffers=%d", min_dec_capture_buffers);
#endif

		err = mDecoder->capture_plane.reqbufs(V4L2_MEMORY_DMABUF, min_dec_capture_buffers);
		if(err < 0) {
			LOGE("%s(%d): mDecoder->capture_plane.reqbufs() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		min_dec_capture_buffers = mDecoder->capture_plane.getNumBuffers();
#if 0 // DEBUG
		LOGD("!!!!! ~min_dec_capture_buffers=%d", min_dec_capture_buffers);
#endif
		err = mDecoder->capture_plane.setStreamStatus(true);
		if(err < 0) {
			LOGE("%s(%d): mDecoder->capture_plane.setStreamStatus() failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		/* Set colorformats for relevant colorspaces. */
		NvBufSurfaceColorFormat pix_format;
		switch(format.fmt.pix_mp.colorspace)
		{
		case V4L2_COLORSPACE_SMPTE170M:
			if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
			{
				LOGD("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)");
				pix_format = NVBUF_COLOR_FORMAT_NV12;
			}
			else
			{
				LOGD("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)");
				pix_format = NVBUF_COLOR_FORMAT_NV12_ER;
			}
			break;
		case V4L2_COLORSPACE_REC709:
			if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
			{
				LOGD("Decoder colorspace ITU-R BT.709 with standard range luma (16-235)");
				pix_format =  NVBUF_COLOR_FORMAT_NV12_709;
			}
			else
			{
				LOGD("Decoder colorspace ITU-R BT.709 with extended range luma (0-255)");
				pix_format = NVBUF_COLOR_FORMAT_NV12_709_ER;
			}
			break;
		case V4L2_COLORSPACE_BT2020:
			{
				LOGD("Decoder colorspace ITU-R BT.2020");
				pix_format = NVBUF_COLOR_FORMAT_NV12_2020;
			}
			break;
		default:
			LOGD("supported colorspace details not available, use default");
			if (format.fmt.pix_mp.quantization == V4L2_QUANTIZATION_DEFAULT)
			{
				LOGD("Decoder colorspace ITU-R BT.601 with standard range luma (16-235)");
				pix_format = NVBUF_COLOR_FORMAT_NV12;
			}
			else
			{
				LOGD("Decoder colorspace ITU-R BT.601 with extended range luma (0-255)");
				pix_format = NVBUF_COLOR_FORMAT_NV12_ER;
			}
			break;
		}

		params.memType = NVBUF_MEM_SURFACE_ARRAY;
		params.width = crop.c.width;
		params.height = crop.c.height;
		params.layout = NVBUF_LAYOUT_BLOCK_LINEAR;
		params.memtag = NvBufSurfaceTag_VIDEO_DEC;

		if (format.fmt.pix_mp.pixelformat  == V4L2_PIX_FMT_NV24M)
			pix_format = NVBUF_COLOR_FORMAT_NV24;
		else if (format.fmt.pix_mp.pixelformat  == V4L2_PIX_FMT_NV24_10LE)
			pix_format = NVBUF_COLOR_FORMAT_NV24_10LE;

	    params.colorFormat = pix_format;

	    mDMABuf_Cap.resize(min_dec_capture_buffers);
	    LOGD("Allocate %d buffers for capturing...", mDMABuf_Cap.size());
		err = NvBufSurf::NvAllocate(&params, mDMABuf_Cap.size(), &mDMABuf_Cap[0]);
		if(err < 0) {
			LOGE("%s(%d): NvBufSurf::NvAllocate failed, err=%d", __FUNCTION__, __LINE__, err);
			break;
		}

		for(int i = 0; i < min_dec_capture_buffers; i++) {
			struct v4l2_buffer v4l2_buf;
			struct v4l2_plane planes[MAX_PLANES];

			memset(&v4l2_buf, 0, sizeof(v4l2_buf));
			memset(planes, 0, sizeof(planes));

			v4l2_buf.index = i;
			v4l2_buf.m.planes = planes;
			v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			v4l2_buf.memory = V4L2_MEMORY_DMABUF;
			v4l2_buf.m.planes[0].m.fd = mDMABuf_Cap[i];

#if 0 // DEBUG
			LOGD("%d: v4l2_buf.m.planes[0].m.fd=%d", i, v4l2_buf.m.planes[0].m.fd);
#endif
			err = mDecoder->capture_plane.qBuffer(v4l2_buf, NULL);
			if(err < 0) {
				LOGE("%s(%d): mDecoder->capture_plane.qBuffer() failed, err=%d", __FUNCTION__, __LINE__, err);
				break;
			}
		}

		mDecodedFormat = format;
	}

	return err;
}