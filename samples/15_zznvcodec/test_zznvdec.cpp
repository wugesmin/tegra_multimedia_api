#include "zznvcodec.h"
#include "ZzLog.h"
#include <unistd.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>

ZZ_INIT_LOG("test_zznvdec");

#define CHUNK_SIZE 4000000

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		!buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		(buffer_ptr[2] == 1))

#define H264_NAL_UNIT_CODED_SLICE  1
#define H264_NAL_UNIT_CODED_SLICE_IDR  5

#define IS_H264_NAL_CODED_SLICE(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE)
#define IS_H264_NAL_CODED_SLICE_IDR(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE_IDR)

#define GET_H265_NAL_UNIT_TYPE(buffer_ptr) ((buffer_ptr[0] & 0x7E) >> 1)

using namespace std;

static int read_decoder_input_nalu(ifstream * stream, char* nalu, int* nalu_size,
		char *parse_buffer, streamsize parse_buffer_size)
{
	// Length is the size of the buffer in bytes
	char *buffer_ptr = (char *) nalu;
	int h265_nal_unit_type;
	char *stream_ptr;
	bool nalu_found = false;

	streamsize bytes_read;
	streamsize stream_initial_pos = stream->tellg();

	stream->read(parse_buffer, parse_buffer_size);
	bytes_read = stream->gcount();

	if (bytes_read == 0)
	{
		return (*nalu_size = 0);
	}

	// Find the first NAL unit in the buffer
	stream_ptr = parse_buffer;
	while ((stream_ptr - parse_buffer) < (bytes_read - 3))
	{
		nalu_found = IS_NAL_UNIT_START(stream_ptr) ||
					IS_NAL_UNIT_START1(stream_ptr);
		if (nalu_found)
		{
			break;
		}
		stream_ptr++;
	}

	// Reached end of buffer but could not find NAL unit
	if (!nalu_found)
	{
		LOGE("%s(%d): Could not read nal unit from file. EOF or file corrupted", __FUNCTION__, __LINE__);
		return -1;
	}

	memcpy(buffer_ptr, stream_ptr, 4);
	buffer_ptr += 4;
	*nalu_size = 4;
	stream_ptr += 4;

	// Copy bytes till the next NAL unit is found
	while ((stream_ptr - parse_buffer) < (bytes_read - 3))
	{
		if (IS_NAL_UNIT_START(stream_ptr) || IS_NAL_UNIT_START1(stream_ptr))
		{
			streamsize seekto = stream_initial_pos +
					(stream_ptr - parse_buffer);
			if(stream->eof())
			{
				stream->clear();
			}
			stream->seekg(seekto, stream->beg);
			return 0;
		}
		*buffer_ptr = *stream_ptr;
		buffer_ptr++;
		stream_ptr++;
		(*nalu_size)++;
	}

	// Reached end of buffer but could not find NAL unit
	LOGE("%s(%d): Could not read nal unit from file. EOF or file corrupted", __FUNCTION__, __LINE__);
	return -1;
}

void _on_video_frame(zznvcodec_video_frame_t* pFrame, int64_t nTimestamp, intptr_t pUser) {
#if 1
	LOGD("%s(%d): %d, frame={%dx%d(%d %p) %dx%d(%d %p) %dx%d(%d %p)}, %.2f\n", __FUNCTION__, __LINE__, pFrame->num_planes,
		pFrame->planes[0].width, pFrame->planes[0].height, pFrame->planes[0].stride, pFrame->planes[0].ptr,
		pFrame->planes[1].width, pFrame->planes[1].height, pFrame->planes[1].stride, pFrame->planes[1].ptr,
		pFrame->planes[2].width, pFrame->planes[2].height, pFrame->planes[2].stride, pFrame->planes[2].ptr,
		nTimestamp / 1000.0);
#endif
}

int main(int argc, char *argv[])
{
	for(int i = 0;;++i) {
		zznvcodec_decoder_t* pDecoder = zznvcodec_decoder_new();

#if 1
		zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_YUV420P;
#endif

#if 0
		zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_NV12;
#endif

		zznvcodec_decoder_set_video_property(pDecoder, 1920, 1080, nPixFmt);
		zznvcodec_pixel_format_t nEncoderPixFmt = ZZNVCODEC_PIXEL_FORMAT_H264;
		zznvcodec_decoder_set_misc_property(pDecoder, ZZNVCODEC_PROP_ENCODER_PIX_FMT, (intptr_t)&nEncoderPixFmt);
		zznvcodec_decoder_register_callbacks(pDecoder, _on_video_frame, 0);

		zznvcodec_decoder_start(pDecoder);

		int ret;
		std::ifstream test_video_file(argv[1], std::ios::binary);
		std::vector<char> nalu_parse_buffer(CHUNK_SIZE);

		int64_t nLastLogTime = 0;
		for(int t = 0;;++t) {
			std::vector<char> nalu_buffer(CHUNK_SIZE);
			int nalu_size;

			ret = read_decoder_input_nalu(&test_video_file, &nalu_buffer[0], &nalu_size, &nalu_parse_buffer[0], nalu_parse_buffer.size());
			if(ret == -1) break;
			if(nalu_size == 0) {
				LOGD("EOF");
				break;
			}

			int64_t nTimestamp = t * 1000000LL / 60;
			if(nTimestamp - nLastLogTime > 1000000LL) {
				LOGD("%d: %.2f", i, nTimestamp / 1000.0);
				nLastLogTime = nTimestamp;
			}

			zznvcodec_decoder_set_video_compression_buffer(pDecoder, (uint8_t*)&nalu_buffer[0], nalu_size, 0, nTimestamp);
			usleep(1000000 / 60);
		}

		zznvcodec_decoder_stop(pDecoder);

		zznvcodec_decoder_delete(pDecoder);
		pDecoder = NULL;
	}

	return 0;
}
