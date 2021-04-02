#pragma once

#ifdef _WIN32
#ifdef VIDEO_STREAMER_EXPORTS
#define VIDEO_STREAMER_API __declspec(dllexport)
#else
#define VIDEO_STREAMER_API __declspec(dllimport)
#endif
#else
#define VIDEO_STREAMER_API
#endif

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#include <stdbool.h>
#endif

#include <iostream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include <vector>
#include <string.h>
#include <iomanip>
#include <exception>
#include <stdexcept>
#include <memory>
#include <functional>
#include "NvCodec/NvDecoder/NvDecoder.h"
#include "NvCodec/NvEncoder/NvEncoderCuda.h"
#include "NvCodec/NvEncoder/NvEncoderOutputInVidMemCuda.h"
#include "Utils/NvCodecUtils.h"
#include "Utils/FFmpegDemuxer.h"
#include "Utils/FFmpegStreamer.h"
#include "Utils/ColorSpace.h"
#include "Utils/Logger.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Common/AppDecUtils.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// #define NVIDIA_ENCODER
// #define APP_ENC_CUDA_CPP
// #define APP_ENC_DEC_CPP
#define NVIDIA_DECODER
#define APP_IN_THREAD_CPP
// #define APP_DEC_LOW_LATENCY_CPP

void AppDec(std::string);

void AppDecLowLatency(std::string);

class VideoStreamer
{

public:
    VideoStreamer(char *, int);
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

void ShowEncoderCapability();

void AppEncCuda();

void AppEncDec();

// This class allocates CUStream.
// It also sets the input and output CUDA stream in the driver, which will be used for pipelining
// pre and post processing CUDA tasks
class NvCUStream
{
public:
    NvCUStream(CUcontext, int, std::unique_ptr<NvEncoderOutputInVidMemCuda> &);
    ~NvCUStream();
    CUstream GetOutputCUStream();
    CUstream GetInputCUStream();

private:
    CUcontext device;
    CUstream inputStream = NULL, outputStream = NULL;
};
// This class computes CRC of encode frame using CUDA kernel
class CRC
{
public:
    CRC(CUcontext, uint32_t);
    ~CRC();
    void GetCRC(NV_ENC_OUTPUT_PTR, CUstream);
    CUdeviceptr GetCRCVidMemPtr();

private:
    CUcontext device;
    CUdeviceptr crcVidMem = 0;
};
// This class dumps the output - CRC and encoded stream, to a file.
// Output is first copied to host buffer and then dumped to a file.
class DumpVidMemOutput
{
public:
    DumpVidMemOutput(CUcontext, uint32_t, char *, bool);
    ~DumpVidMemOutput();
    void DumpOutputToFile(CUdeviceptr, CUdeviceptr, std::ofstream &, uint32_t);

private:
    CUcontext device;
    uint32_t bfrSize;
    uint8_t *pHostMemEncOp = NULL;
    uint32_t *pHostMemCRC = NULL;
    bool bCRC;
    std::string crcFile;
    std::ofstream fpCRCOut;
};

template <class EncoderClass>

void InitializeEncoder(EncoderClass &, NvEncoderInitParam, NV_ENC_BUFFER_FORMAT);

void EncodeCuda(int, int, NV_ENC_BUFFER_FORMAT, NvEncoderInitParam, CUcontext, std::ifstream &, std::ofstream &);

void EncodeCudaOpInVidMem(int, int, NV_ENC_BUFFER_FORMAT, NvEncoderInitParam, CUcontext, std::ifstream &, std::ofstream &, char *, int32_t);

enum OutputFormat
{
    native = 0,
    bgra,
    bgra64
};

std::vector<std::string> vstrOutputFormatName = {
    "native", "bgra", "bgra64"};

void EncodeProc(CUdevice, int, int, NV_ENC_BUFFER_FORMAT, NvEncoderInitParam *, bool, const char *, const char *, std::exception_ptr &);

void DecodeProc(CUdevice, const char *, OutputFormat, const char *, std::exception_ptr &);

extern "C" VIDEO_STREAMER_API VideoStreamer *VideoStreamer_new(VideoStreamer *, char *, int);
extern "C" VIDEO_STREAMER_API void update(VideoStreamer *);
// extern "C" VIDEO_STREAMER_API char *read(VideoStreamer *);
extern "C" VIDEO_STREAMER_API bool status(VideoStreamer *);
extern "C" VIDEO_STREAMER_API int getWidth(VideoStreamer *);
extern "C" VIDEO_STREAMER_API int getHeight(VideoStreamer *);
extern "C" VIDEO_STREAMER_API void kill_stream(VideoStreamer *);

#ifdef __cplusplus
extern "C"
{
#endif
    VideoStreamer *VideoStreamer_new(VideoStreamer *stream, char *srcStr, int iGpu)
    {
        return new VideoStreamer(srcStr, iGpu);
    }
    void update(VideoStreamer *stream) { stream->update(); }
    // char *read(VideoStreamer *stream) { return (char *)stream->read(); }
    bool status(VideoStreamer *stream) { return stream->status(); }
    int getWidth(VideoStreamer *stream) { return stream->getWidth(); };
    int getHeight(VideoStreamer *stream) { return stream->getHeight(); };
    void kill_stream(VideoStreamer *stream)
    {
        if (stream)
        {
            delete stream;
            stream = nullptr;
        }
    }
#ifdef __cplusplus
}
#endif
