#include "zznvcodec.h"
#include "ZzLog.h"
#include <unistd.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>

#define OutputFile

ZZ_INIT_LOG("test_zznvdec");

#define CHUNK_SIZE 4000000

#define IS_NAL_UNIT_START(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		!buffer_ptr[2] && (buffer_ptr[3] == 1))

#define IS_NAL_UNIT_START1(buffer_ptr) (!buffer_ptr[0] && !buffer_ptr[1] && \
		(buffer_ptr[2] == 1))

#define H264_NAL_UNIT_CODED_SLICE  1
#define H264_NAL_UNIT_CODED_SLICE_IDR  5

#define IVF_FILE_HDR_SIZE   32
#define IVF_FRAME_HDR_SIZE  12

#define IS_H264_NAL_CODED_SLICE(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE)
#define IS_H264_NAL_CODED_SLICE_IDR(buffer_ptr) ((buffer_ptr[0] & 0x1F) == H264_NAL_UNIT_CODED_SLICE_IDR)

#define GET_H265_NAL_UNIT_TYPE(buffer_ptr) ((buffer_ptr[0] & 0x7E) >> 1)

using namespace std;

FILE *fp;

int vp9_file_header_flag = 0;

// AV1 reader
static int
read_vpx_decoder_input_chunk(ifstream * stream, char* nalu, int* nalu_size)
{
    //ifstream *stream = ctx->in_file[0];
    int Framesize;
    char *bitstreambuffer = nalu;
    if (vp9_file_header_flag == 0)
    {
        stream->read(nalu, IVF_FILE_HDR_SIZE);
        if (stream->gcount() !=  IVF_FILE_HDR_SIZE)
        {
            LOGE("Couldn't read IVF FILE HEADER");
            return -1;
        }
        if (!((bitstreambuffer[0] == 'D') && (bitstreambuffer[1] == 'K') &&
                    (bitstreambuffer[2] == 'I') && (bitstreambuffer[3] == 'F')))
        {
            LOGE("It's not a valid IVF file \n");
            return -1;
        }
        LOGD("It's a valid IVF file");
        vp9_file_header_flag = 1;
    }
    stream->read(nalu, IVF_FRAME_HDR_SIZE);

    if (!stream->gcount())
    {
        LOGD("End of stream");
        return 0;
    }

    if (stream->gcount() != IVF_FRAME_HDR_SIZE)
    {
        LOGE("Couldn't read IVF FRAME HEADER");
        return -1;
    }
    Framesize = (bitstreambuffer[3]<<24) + (bitstreambuffer[2]<<16) +
        (bitstreambuffer[1]<<8) + bitstreambuffer[0];
    *nalu_size = Framesize;
    stream->read(nalu, Framesize);
    if (stream->gcount() != Framesize)
    {
        LOGE("Couldn't read Framesize");
        return -1;
    }
    LOGD("read_vpx_decoder_input_chunk end  %d",*nalu_size);
    return 0;
}

// H264 / H265 reader
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
		LOGW("%s(%d): Could not read nal unit from file. EOF or file corrupted", __FUNCTION__, __LINE__);
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
	LOGD("%s(%d): %d, frame={%dx%d(%d %p) %dx%d(%d %p) %dx%d(%d %p)}, %.2f\n", __FUNCTION__, __LINE__, pFrame->num_planes,
		pFrame->planes[0].width, pFrame->planes[0].height, pFrame->planes[0].stride, pFrame->planes[0].ptr,
		pFrame->planes[1].width, pFrame->planes[1].height, pFrame->planes[1].stride, pFrame->planes[1].ptr,
		pFrame->planes[2].width, pFrame->planes[2].height, pFrame->planes[2].stride, pFrame->planes[2].ptr,
		nTimestamp / 1000.0);
#ifdef OutputFile			
		for (int k = 0 ; k< pFrame->num_planes ; k++)
		{
			for (int i =0 ; i< pFrame->planes[k].height ; i++)
			{
				fwrite(pFrame->planes[k].ptr + i * pFrame->planes[k].stride, 1, pFrame->planes[k].stride, fp);
			}
		} 
#endif		
}

int main(int argc, char *argv[])
{
	int width = 3840;
	int height = 2160;
	unsigned char *pOutBuffer = NULL;
#ifdef OutputFile	
	fp = fopen("test_av1_2.yuv","w");
#endif	
	for(int i = 0;i < 1;++i) {
		zznvcodec_decoder_t* pDecoder = zznvcodec_decoder_new();

#if 0
		zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_NV24;
#else
		zznvcodec_pixel_format_t nPixFmt = ZZNVCODEC_PIXEL_FORMAT_NV12;
#endif

		zznvcodec_decoder_set_video_property(pDecoder, width, height, nPixFmt);
		zznvcodec_pixel_format_t nEncoderPixFmt = ZZNVCODEC_CODEC_TYPE_AV1;
			
		zznvcodec_decoder_set_misc_property(pDecoder, ZZNVCODEC_PROP_ENCODER_PIX_FMT, (intptr_t)&nEncoderPixFmt);
#ifndef DIRECT_OUTPUT	
		zznvcodec_decoder_register_callbacks(pDecoder, _on_video_frame, 0);
#endif
		zznvcodec_decoder_start(pDecoder);

#if (defined OutputFile) && (defined DIRECT_OUTPUT)		
		pOutBuffer = (unsigned char*) malloc(width*height*3 * sizeof(unsigned char));			
#endif	
		int ret;
		std::ifstream test_video_file(argv[1], std::ios::binary);
		std::vector<char> nalu_parse_buffer(CHUNK_SIZE);

		int64_t nLastLogTime = 0;
		for(int t = 0;t < 100;++t) {
			std::vector<char> nalu_buffer(CHUNK_SIZE);
			int nalu_size;

			if ((nEncoderPixFmt == ZZNVCODEC_CODEC_TYPE_H264) || (nEncoderPixFmt == ZZNVCODEC_CODEC_TYPE_H265)) {
				ret = read_decoder_input_nalu(&test_video_file, &nalu_buffer[0], &nalu_size, &nalu_parse_buffer[0], nalu_parse_buffer.size());
				if(ret == -1) break;
				if(nalu_size == 0) {
					LOGD("EOF");
					break;
				}
			} 
			else if (nEncoderPixFmt == ZZNVCODEC_CODEC_TYPE_AV1) {
				ret = read_vpx_decoder_input_chunk(&test_video_file, &nalu_buffer[0], &nalu_size);
				if(ret == -1) break;
				if(nalu_size == 0) {
					LOGD("EOF");
					break;
				}				
			}

			int64_t nTimestamp = t * 1000000LL / 60;
			int64_t nOutSize = 0;
			int64_t nOutTimeStamp = 0;
			
			if(nTimestamp - nLastLogTime > 1000000LL) {
				LOGD("%d: %.2f", i, nTimestamp / 1000.0);
				nLastLogTime = nTimestamp;
			}
			LOGD("zznvcodec_decoder_set_video_compression_buffer Begin");
			zznvcodec_decoder_set_video_compression_buffer(pDecoder, (uint8_t*)&nalu_buffer[0], nalu_size, 0, nTimestamp, pOutBuffer, &nOutSize, &nOutTimeStamp);

#if (defined OutputFile) && (defined DIRECT_OUTPUT)		
			// Direct Output
			if (nOutSize != 0) {
				LOGD("%s(%d): ,outsize: %d timestamp: %.2f\n", __FUNCTION__, __LINE__, nOutSize, nOutTimeStamp / 1000.0);	
				fwrite(pOutBuffer, 1, nOutSize, fp);
			}			
#endif
			usleep(1000000 / 60);
		}

#if (defined OutputFile) && (defined DIRECT_OUTPUT)	
		// Flush Frame
		while (1)
		{
			int64_t nOutSize = 0;
			int64_t nOutTimeStamp = 0;		
			zznvcodec_decoder_set_video_compression_buffer(pDecoder, NULL, 0, 0, 0, pOutBuffer, &nOutSize, &nOutTimeStamp);	
			// Direct Output
			if (nOutSize != 0) {
				LOGD("%s(%d): , %.2f\n", __FUNCTION__, __LINE__, nOutTimeStamp / 1000.0);	
				fwrite(pOutBuffer, 1, nOutSize, fp);
			}
			else
				break;
		}
		free(pOutBuffer);	
#endif
		
		zznvcodec_decoder_stop(pDecoder);

		zznvcodec_decoder_delete(pDecoder);
		pDecoder = NULL;
	}
#ifdef OutputFile	
	fclose(fp);
#endif
	return 0;
}
