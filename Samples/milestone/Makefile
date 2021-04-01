################################################################################
#
# Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
#
################################################################################

include common.mk

NVCCFLAGS := $(CCFLAGS)

LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart -lnvcuvid -pthread
LDFLAGS += $(shell pkg-config --libs libavcodec libavutil libavformat opencv)
INCLUDES += $(shell pkg-config --cflags libavcodec libavutil libavformat opencv)

# Target rules
all: build

build: app

NvDecoder.o: NvCodec/NvDecoder/NvDecoder.cpp NvCodec/NvDecoder/NvDecoder.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

main.o: main.cpp main.hpp NvCodec/NvDecoder/NvDecoder.h \
		Utils/NvCodecUtils.h Utils/FFmpegDemuxer.h \
		Common/AppDecUtils.h Utils/Logger.h
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

app: main.o NvDecoder.o
	$(GCC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -rf app main.o NvDecoder.o

