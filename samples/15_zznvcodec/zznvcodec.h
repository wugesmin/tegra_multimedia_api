#ifndef __ZZNVCODEC_H__
#define __ZZNVCODEC_H__

#include <sys/types.h>
#include <sys/socket.h>
#include <stdint.h>

#define ZZNVCODEC_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif

struct zznvcodec_decoder_t;
struct zznvcodec_encoder_t;

enum zznvcodec_consts_t {
	ZZNVCODEC_MAX_PLANES = 3,
};

enum zznvcodec_props_t {
	ZZNVCODEC_PROP_ENCODER_PIX_FMT,		// zznvcodec_pixel_format_t
	ZZNVCODEC_PROP_BITRATE,				// int
	ZZNVCODEC_PROP_PROFILE,				// int
	ZZNVCODEC_PROP_LEVEL,				// int
	ZZNVCODEC_PROP_RATECONTROL,			// int
	ZZNVCODEC_PROP_IDRINTERVAL,			// int
	ZZNVCODEC_PROP_IFRAMEINTERVAL,		// int
	ZZNVCODEC_PROP_FRAMERATE,			// int[2] (num/deno)
};

struct zznvcodec_video_plane_t {
	int width;
	int height;
	uint8_t* ptr;
	int stride;
};

struct zznvcodec_video_frame_t {
	int num_planes;
	zznvcodec_video_plane_t planes[ZZNVCODEC_MAX_PLANES];
};

enum zznvcodec_pixel_format_t {
	ZZNVCODEC_PIXEL_FORMAT_UNKNOWN = -1,
	ZZNVCODEC_PIXEL_FORMAT_NV12,
	ZZNVCODEC_PIXEL_FORMAT_YUV420P,
	ZZNVCODEC_PIXEL_FORMAT_YUYV422,
	ZZNVCODEC_PIXEL_FORMAT_H264,
	ZZNVCODEC_PIXEL_FORMAT_H265,
};

typedef void (*zznvcodec_decoder_on_video_frame_t)(zznvcodec_video_frame_t* pFrame, int64_t nTimestamp, intptr_t pUser);
typedef void (*zznvcodec_encoder_on_video_packet_t)(unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp, intptr_t pUser);

ZZNVCODEC_API zznvcodec_decoder_t* zznvcodec_decoder_new();
ZZNVCODEC_API void zznvcodec_decoder_delete(zznvcodec_decoder_t* pThis);

ZZNVCODEC_API void zznvcodec_decoder_set_video_property(zznvcodec_decoder_t* pThis, int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat);
ZZNVCODEC_API void zznvcodec_decoder_set_misc_property(zznvcodec_decoder_t* pThis, int nProperty, intptr_t pValue);
ZZNVCODEC_API void zznvcodec_decoder_register_callbacks(zznvcodec_decoder_t* pThis, zznvcodec_decoder_on_video_frame_t pCB, intptr_t pUser);

ZZNVCODEC_API int zznvcodec_decoder_start(zznvcodec_decoder_t* pThis);
ZZNVCODEC_API void zznvcodec_decoder_stop(zznvcodec_decoder_t* pThis);

ZZNVCODEC_API void zznvcodec_decoder_set_video_compression_buffer(zznvcodec_decoder_t* pThis, unsigned char* pBuffer, int nSize, int nFlags, int64_t nTimestamp);

ZZNVCODEC_API zznvcodec_encoder_t* zznvcodec_encoder_new();
ZZNVCODEC_API void zznvcodec_encoder_delete(zznvcodec_encoder_t* pThis);

ZZNVCODEC_API void zznvcodec_encoder_set_video_property(zznvcodec_encoder_t* pThis, int nWidth, int nHeight, zznvcodec_pixel_format_t nFormat);
ZZNVCODEC_API void zznvcodec_encoder_set_misc_property(zznvcodec_encoder_t* pThis, int nProperty, intptr_t pValue);
ZZNVCODEC_API void zznvcodec_encoder_register_callbacks(zznvcodec_encoder_t* pThis, zznvcodec_encoder_on_video_packet_t pCB, intptr_t pUser);

ZZNVCODEC_API int zznvcodec_encoder_start(zznvcodec_encoder_t* pThis);
ZZNVCODEC_API void zznvcodec_encoder_stop(zznvcodec_encoder_t* pThis);

ZZNVCODEC_API void zznvcodec_encoder_set_video_uncompression_buffer(zznvcodec_encoder_t* pThis, zznvcodec_video_frame_t* pFrame, int64_t nTimestamp);

#ifdef __cplusplus
}
#endif

#endif // __ZZNVCODEC_H__
