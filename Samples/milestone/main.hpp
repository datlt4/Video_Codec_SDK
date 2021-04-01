#pragma once

#ifdef VIDEO_STREAMER_EXPORTS
#define VIDEO_STREAMER_API __declspec(dllexport)
#else
#define VIDEO_STREAMER_API __declspec(dllimport)
#endif

#include <iostream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/ColorSpace.h"
#include "Common/AppDecUtils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <string.h>

void AppDec(std::string);
void AppDecLowLatency(std::string);
