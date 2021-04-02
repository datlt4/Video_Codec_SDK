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

class VideoStreamer
{

public:
    VideoStreamer(std::string, int);
    ~VideoStreamer();

    void update();
    uchar *read();
    void read(bool &, cv::Mat &);
    bool status();
    int getWidth();
    int getHeight();

private:
    std::string *srcStr;
    int iGpu = 0;

    volatile bool _status = false;
    int frame_width, frame_height;
    cv::Mat mat_bgr;
};

// extern "C" VIDEO_STREAMER_API VideoStreamer *VideoStreamer_new();
// extern "C" VIDEO_STREAMER_API void init(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API void newStrSource(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API void update(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API const char *read(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API bool status(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API void kill_stream(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API bool empty(VideoStreamer *);

// extern "C"
// {
//     VideoStreamer *VideoStreamer_new() { return new VideoStreamer(); }
//     void init(VideoStreamer *stream) { stream->init(); }
//     void newStrSource(VideoStreamer *stream) { stream->newStrSource(); }
//     void update(VideoStreamer *stream) { stream->update(); }
//     const char *read(VideoStreamer *stream) { return stream->read(); }
//     bool status(VideoStreamer *stream) { return stream->status(); }
//     bool empty(VideoStreamer *stream) { return stream->empty(); }
//     void kill_stream(VideoStreamer *stream)
//     {
//         if (stream)
//         {
//             delete stream;
//             stream = nullptr;
//         }
//     }
// }
