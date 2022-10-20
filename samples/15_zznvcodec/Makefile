###############################################################################
#
# Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

include ../Rules.mk

ZZNVCODEC_SRCS := \
	ZzLog.cpp \
	zznvdec.cpp \
	zznvenc.cpp \
	$(wildcard $(CLASS_DIR)/*.cpp)

ZZNVCODEC_OBJS := $(ZZNVCODEC_SRCS:.cpp=.o)
ZZNVCODEC_LIB := zznvcodec

VIDEO_DECODE_APP := video_decode
VIDEO_DECODE_SRCS := \
	video_decode_csvparser.cpp \
	video_decode_main.cpp \
	$(wildcard $(CLASS_DIR)/*.cpp)
VIDEO_DECODE_OBJS := $(VIDEO_DECODE_SRCS:.cpp=.o)

VIDEO_ENCODE_APP := video_encode
VIDEO_ENCODE_SRCS := \
	video_encode_csvparser.cpp \
	video_encode_main.cpp \
	$(wildcard $(CLASS_DIR)/*.cpp)
VIDEO_ENCODE_OBJS := $(VIDEO_ENCODE_SRCS:.cpp=.o)

TEST_ZZNVDEC_SRCS := \
	ZzLog.cpp \
	test_zznvdec.cpp
TEST_ZZNVDEC_OBJS := $(TEST_ZZNVDEC_SRCS:.cpp=.o)
TEST_ZZNVDEC_APP := test_zznvdec

TEST_ZZNVENC_SRCS := \
	ZzLog.cpp \
	test_zznvenc.cpp
TEST_ZZNVENC_OBJS := $(TEST_ZZNVENC_SRCS:.cpp=.o)
TEST_ZZNVENC_APP := test_zznvenc

all: $(ZZNVCODEC_LIB) $(VIDEO_ENCODE_APP) $(VIDEO_DECODE_APP) $(TEST_ZZNVDEC_APP) $(TEST_ZZNVENC_APP)

clean:
	$(AT)rm -rf $(VIDEO_DECODE_APP) $(VIDEO_DECODE_OBJS) $(VIDEO_ENCODE_APP) $(VIDEO_ENCODE_OBJS) \
	$(TEST_ZZNVDEC_APP) $(TEST_ZZNVDEC_OBJS) \
	$(TEST_ZZNVENC_APP) $(TEST_ZZNVENC_OBJS) $(ZZNVCODEC_OBJS) *.so

%.o: %.cpp
	@echo "Compiling: $<"
	$(CPP) $(CPPFLAGS) -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -c $<

$(CLASS_DIR)/%.o: $(CLASS_DIR)/%.cpp
	@echo "Compiling NV classes: $<"
	$(CPP) $(CPPFLAGS) -fPIC -c $< -o $(patsubst %.cpp,%.o,$<)

$(ZZNVCODEC_LIB): $(ZZNVCODEC_OBJS)
	@echo "Compiling: $@"
	$(CPP) $(CPPFLAGS) -shared -o lib$(ZZNVCODEC_LIB).so $(ZZNVCODEC_OBJS) $(LDFLAGS) -Wl,--version-script=zznvcodec.map -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps

$(VIDEO_DECODE_APP): $(VIDEO_DECODE_OBJS)
	@echo "Linking: $@"
	$(CPP) $(CPPFLAGS) -o $@ $(VIDEO_DECODE_OBJS) $(CPPFLAGS) $(LDFLAGS)

$(VIDEO_ENCODE_APP): $(VIDEO_ENCODE_OBJS)
	@echo "Linking: $@"
	$(CPP) $(CPPFLAGS) -o $@ $(VIDEO_ENCODE_OBJS) $(CPPFLAGS) $(LDFLAGS)

$(TEST_ZZNVDEC_APP): $(ZZNVCODEC_LIB) $(TEST_ZZNVDEC_OBJS)
	@echo "Linking: $@"
	$(CPP) $(CPPFLAGS) -o $@ $(TEST_ZZNVDEC_OBJS) $(CPPFLAGS) $(LDFLAGS) -L. -l$(ZZNVCODEC_LIB)

$(TEST_ZZNVENC_APP): $(ZZNVCODEC_LIB) $(TEST_ZZNVENC_OBJS)
	@echo "Linking: $@"
	$(CPP) $(CPPFLAGS) -o $@ $(TEST_ZZNVENC_OBJS) $(CPPFLAGS) $(LDFLAGS) -L. -l$(ZZNVCODEC_LIB) -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps
