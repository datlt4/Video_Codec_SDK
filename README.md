**Environment**
> OS: Ubuntu 18.04.5 LTS x86_64<br>CUDA: 10.0<br>SDK: Video_Codec_SDK_9.1.23<br>
FFmpeg: N-101443-g74b5564fb5<br>gcc 7<br>Python: 3.6

[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Reference-com_logo.svg/1280px-Reference-com_logo.svg.png" height="15">](https://github.com/LuongTanDat/Video_Codec_SDK/blob/v9.1.23/doc/NVENC_VideoEncoder_API_ProgGuide.pdf)
[<img src= "https://aimansoliman.files.wordpress.com/2020/02/github_owler_20180612_070358_original.png" height="25">](https://github.com/LuongTanDat/Video_Codec_SDK/tree/v9.1.23/Samples/milestone)


# <span style="color:red">♡ INSTALL ENVIRONMENT ♥

- [Download Ubuntu](https://releases.ubuntu.com/18.04/)
- [Install CUDA](https://github.com/LuongTanDat/WLINUX/blob/master/Install_new_linux.md#install-cuda---allows-us-a-way-to-write-code-for-gpus-install-cuda-100---101)
- [Install FFmpeg with Nvidia Accelator](https://github.com/LuongTanDat/WLINUX/blob/master/Install_new_linux.md#install-ffmpeg-with-nvidia-accelator)
- [Video Codec SDK Archive](https://developer.nvidia.com/video-codec-sdk-archive)

# <span style="color:red">♢ DECODING ANY VIDEO CONTENT USING NVDECODE API ♦

## Step <a name="introduction"></a>
1. [Create a `CUDA` context.](#create_a_cuda_context)
2. [Query the `decode capabilities` of the hardware decoder.](#query_the_decode_capabilities)
3. [Create the decoder instance(s).](#create_decoder_instance_and_ffmpeg_demuxer)
4. [De-Mux the content (like .mp4). `FFMPEG`.](#create_decoder_instance_and_ffmpeg_demuxer)
5. Parse the video `bitstream` using third party parser like `FFMPEG`.
6. [Kick off the Decoding using NVDECODE API.](#start_decoding_and_obtain_YUV)
7. [Obtain the `decoded YUV` for further processing.](#start_decoding_and_obtain_YUV)
8. Query the `status` of the decoded frame.
9. Depending on the `decoding status`, use the decoded output for further processing like `rendering`, `inferencing`, `postprocessing` etc.
10. [Convert decoded `YUV` surface to `RGB`.](#full_source_code)
11. Destroy the decoder instance(s) after the completion of decoding process.
12. Destroy the `CUDA` context.

**_Note:_**

- `CUDA context`: holds all management data to control and use the devices. For instance, it holds the list of allocated memory, the loaded modules that contain device code, the mapping between CPU and GPU memory.

- `Bitstream`: data found in a stream of bits used in digital communication or data storage application.

- `BSD`: bit stream decoder.

- All `NVDECODE API` action are exposed in two header-file `cuviddec.h`, `nvcuvid.h` and a static library `libnvcuvid.so`.

- NVIDIA GPU-accelerated decoder pipeline consists of three major components: `Demultiplexing` _(FFMPEG)_, `parsing` _(pure SW FFMPEG)_, `decoding` _(GPU-acceleration)_.

- Before creating decoder, we should query capabilities of GPU to ensure that required functional is supported.

## Create a `CUDA` context <a name="create_a_cuda_context"></a>

- `Common/AppDecUtils.h`
<details>
  <summary>Click to expand!</summary>

  ```cpp
  static void createCudaContext(CUcontext* cuContext, int iGpu, unsigned int flags)
  {
      CUdevice cuDevice = 0;
      ck(cuDeviceGet(&cuDevice, iGpu));
      char szDeviceName[80];
      ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
      std::cout << "GPU in use: " << szDeviceName << std::endl;
      ck(cuCtxCreate(cuContext, flags, cuDevice));
  }
  ```
  - `cuda.h -> cuDeviceGet(CUdevice *device, int ordinal)`: Returns in *device a device handle given an ordinal in the range.
  - `cuda.h -> cuDeviceGetName(char *name, int len, CUdevice dev)`: Returns an identifer string for the device.
  - `cuda.h -> cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)`: Creates a new CUDA context and associates it with the calling thread.

</details>

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"

  simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

  int main()
  {
      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));

      CUcontext cuContext = NULL;
      int iGpu = 0;
      createCudaContext(&cuContext, iGpu, 0);
      ck(cuCtxDestroy(cuContext));
      return 0;
  }
  ```
  - `cuda.h -> cuInit(unsigned int Flags)`: Initializes the driver API and must be called before any other function from the driver API.
  - `cuda.h -> cuDeviceGetCount(int *count)`: Returns an identifer string for the device.
  - `cuda.h -> cuCtxDestroy(CUcontext ctx)`: Destroy a CUDA context.
</details>

## Query the `decode capabilities` of the hardware decoder <a name="query_the_decode_capabilities"></a>
- `Common/AppDecUtils.h`
<details>
  <summary>Click to expand!</summary>

  ```cpp
  static void ShowDecoderCapability()
  {
      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));
      std::cout << "Decoder Capability" << std::endl << std::endl;
      const char *aszCodecName[] = {"JPEG", "MPEG1", "MPEG2", "MPEG4", "H264", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "VC1", "VP8", "VP9", "VP9", "VP9"};
      const char *aszChromaFormat[] = { "4:0:0", "4:2:0", "4:2:2", "4:4:4" };
      char strOutputFormats[64];
      cudaVideoCodec aeCodec[] = { cudaVideoCodec_JPEG, cudaVideoCodec_MPEG1, cudaVideoCodec_MPEG2, cudaVideoCodec_MPEG4, cudaVideoCodec_H264, cudaVideoCodec_HEVC,
          cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_VC1, cudaVideoCodec_VP8,
          cudaVideoCodec_VP9, cudaVideoCodec_VP9, cudaVideoCodec_VP9 };
      int anBitDepthMinus8[] = {0, 0, 0, 0, 0, 0, 2, 4, 0, 2, 4, 0, 0, 0, 2, 4};

      cudaVideoChromaFormat aeChromaFormat[] = { cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
          cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_444, cudaVideoChromaFormat_444,
          cudaVideoChromaFormat_444, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420 };

      for (int iGpu = 0; iGpu < nGpu; iGpu++)
      {
          CUcontext cuContext = NULL;
          CUdevice cuDevice = 0;
          ck(cuDeviceGet(&cuDevice, iGpu));
          char szDeviceName[80];
          ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
          std::cout << "GPU in use: " << szDeviceName << std::endl;
          ck(cuCtxCreate(&cuContext, 0, cuDevice));

          std::cout << "Codec  BitDepth  ChromaFormat  Supported  MaxWidth  MaxHeight  MaxMBCount  MinWidth  MinHeight  SurfaceFormat" << std::endl;

          for (int i = 0; i < sizeof(aeCodec) / sizeof(aeCodec[0]); i++)
          {
              CUVIDDECODECAPS decodeCaps = {};
              decodeCaps.eCodecType = aeCodec[i];
              decodeCaps.eChromaFormat = aeChromaFormat[i];
              decodeCaps.nBitDepthMinus8 = anBitDepthMinus8[i];

              cuvidGetDecoderCaps(&decodeCaps);

              strOutputFormats[0] = '\0';
              getOutputFormatNames(decodeCaps.nOutputFormatMask, strOutputFormats);

              // setw() width = maximum_width_of_string + 2 spaces
              std::cout << std::left << std::setw(std::string("Codec").length() + 2) << aszCodecName[i] <<
                          std::setw(std::string("BitDepth").length() + 2) << decodeCaps.nBitDepthMinus8 + 8 <<
                          std::setw(std::string("ChromaFormat").length() + 2) << aszChromaFormat[decodeCaps.eChromaFormat] <<
                          std::setw(std::string("Supported").length() + 2) << (int)decodeCaps.bIsSupported <<
                          std::setw(std::string("MaxWidth").length() + 2) << decodeCaps.nMaxWidth <<
                          std::setw(std::string("MaxHeight").length() + 2) << decodeCaps.nMaxHeight <<
                          std::setw(std::string("MaxMBCount").length() + 2) << decodeCaps.nMaxMBCount <<
                          std::setw(std::string("MinWidth").length() + 2) << decodeCaps.nMinWidth <<
                          std::setw(std::string("MinHeight").length() + 2) << decodeCaps.nMinHeight <<
                          std::setw(std::string("SurfaceFormat").length() + 2) << strOutputFormats << std::endl;
          }
          std::cout << std::endl;
          ck(cuCtxDestroy(cuContext));
      }
  }
  ```
  - `cuda.h -> cuInit(unsigned int Flags)`: Initializes the driver API and must be called before any other function from the driver API.
  - `cuda.h -> cuDeviceGetCount(int *count)`: Returns an identifer string for the device.
  - `cuda.h -> cuDeviceGet(CUdevice *device, int ordinal)`: Returns in *device a device handle given an ordinal in the range.
  - `cuda.h -> cuDeviceGetName(char *name, int len, CUdevice dev)`: Returns an identifer string for the device.
  - `cuda.h -> cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)`: Creates a new CUDA context and associates it with the calling thread.
  - `cuda.h -> cuCtxDestroy(CUcontext ctx)`: Destroy a CUDA context.
  - `cuviddec.h -> cuvidGetDecoderCaps(CUVIDDECODECAPS *pdc)`: Queries decode capabilities of NVDEC-HW based on CodecType, ChromaFormat and BitDepthMinus8 parameters.
</details>

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"

  simplelogger::Logger *logger =   simplelogger::LoggerFactory::CreateConsoleLogger();

  int main()
  {
      ShowDecoderCapability();
      return 0;
  }
  ```
  - `Common/AppDecUtils.h -> ShowDecoderCapability()`: Show decoder capabilities of all GPU.
</details>

## Create decoder instance and FFmpeg Demuxer <a name="create_decoder_instance_and_ffmpeg_demuxer"></a>

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"

  simplelogger::Logger *logger =   simplelogger::LoggerFactory::CreateConsoleLogger();

  int main()
  {
      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));
      CUcontext cuContext = NULL;
      int iGpu = 0;

      std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";
      createCudaContext(&cuContext, iGpu, 0);
      FFmpegDemuxer demuxer(szInFilePath.c_str());
      NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false, false);
      LOG(INFO) << dec.GetVideoInfo();
      std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};
      LOG(INFO) << "Output format: " << aszDecodeOutFormat[dec.GetOutputFormat()];
      ck(cuCtxDestroy(cuContext));

      return 0;
  }
  ```

  - `Utils/FFmpegDemuxer.h -> class FFmpegDemuxer`: class provides functionality for stream demuxing.
  - `NvCodec/NvDecoder/NvDecoder.h -> class NvDecoder`: Base class for decoder interface.
</details>

## Kick off the decoding and obtain decoded YUV for further processing <a name="start_decoding_and_obtain_YUV"></a>

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"

  simplelogger::Logger *logger =   simplelogger::LoggerFactory::CreateConsoleLogger();

  int main()
  {
      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));

      CUcontext cuContext = NULL;
      int iGpu = 0;

      createCudaContext(&cuContext, iGpu, 0);
      std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";
      FFmpegDemuxer demuxer(szInFilePath.c_str());
      NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false, false);

      int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
      uint8_t *pVideo = NULL, **ppFrame;
      std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};

      do
      {
          demuxer.Demux(&pVideo, &nVideoBytes);
          dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
          if (!nFrame && nFrameReturned)
          {
              LOG(INFO) << dec.GetVideoInfo();
              LOG(INFO) << "Output format: " << aszDecodeOutFormat[dec.GetOutputFormat()];
          }
      } while (nVideoBytes);

      ck(cuCtxDestroy(cuContext));
      LOG(INFO) << "End of process";
  }
  ```
</details>

## Full source code <a name="full_source_code"></a>

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"
  #include <opencv2/highgui.hpp>
  #include <opencv2/imgproc.hpp>
  
  simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();
  
  int main()
  {
      ShowDecoderCapability();
      std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";

      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));
      CUcontext cuContext = NULL;
      int iGpu = 0;

      createCudaContext(&cuContext, iGpu, 0);
      FFmpegDemuxer demuxer(szInFilePath.c_str());
      NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false, false);
      int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
      uint8_t *pVideo = NULL, **ppFrame;
      cv::Mat mat_bgr;
      std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};
      cv::namedWindow("a", cv::WINDOW_NORMAL);
      do
      {
          demuxer.Demux(&pVideo, &nVideoBytes);
          dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
          if (!nFrame && nFrameReturned)
          {
              LOG(INFO) << dec.GetVideoInfo();
              LOG(INFO) << "Output format: " << aszDecodeOutFormat[dec.GetOutputFormat()];
          }

          if (nFrameReturned < 1)
              continue;
          nFrame += nFrameReturned;
          cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 1.5, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
          cv::Mat mat_rgb = cv::Mat(dec.GetHeight(), dec.GetWidth(), CV_8UC3);
          cv::cvtColor(mat_yuv, mat_rgb, cv::COLOR_YUV2BGR_NV21);
          cv::cvtColor(mat_rgb, mat_bgr, cv::COLOR_RGB2BGR);
  
          cv::imshow("a", mat_bgr);
          char c = (char)cv::waitKey(25);
          if (c == 27)
              break;
      } while (nVideoBytes);

      ck(cuCtxDestroy(cuContext));
      LOG(INFO) << "End of process";

      return 0;
  }
  ```
</details>

# <span style="color:red">♧ LOW LATENCY ♣

This sample application demonstrates low latency decoding feature. This feature helps to get output frame as soon as it is decoded without any delay. The feature will work for streams having [`I` and `P` frames](#video_compression_picture_types) only.

[**_[cited]_**](https://en.wikipedia.org/wiki/Video_compression_picture_types)<a name="video_compression_picture_types"></a>
In the field of video compression a video frame is compressed using different algorithms with different advantages and disadvantages, centered mainly around amount of data compression. These different algorithms for video frames are called picture types or frame types. The three major picture types used in the different video algorithms are `I`, `P` and `B`. They are different in the following characteristics:

- `I‑frames` are the least compressible but don't require other video frames to decode.
- `P‑frames` can use data from previous frames to decompress and are more compressible than `I‑frames`.
- `B‑frames` can use both previous and forward frames for data reference to get the highest amount of data compression.

## Source

- `main.cpp`

<details>
  <summary>Click to expand!</summary>

  ```cpp
  #include <iostream>
  #include <algorithm>
  #include <thread>
  #include <cuda.h>
  #include "NvCodec/NvDecoder/NvDecoder.h"
  #include "Utils/NvCodecUtils.h"
  #include "Utils/FFmpegDemuxer.h"
  #include "Common/AppDecUtils.h"
  #include <opencv2/highgui.hpp>
  #include <opencv2/imgproc.hpp>
  #include <vector>
  #include <string.h>
  
  int main()
  {
      ShowDecoderCapability();
      std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";
      try
      {
          ck(cuInit(0));
          int nGpu = 0;
          int iGpu = 0;
          ck(cuDeviceGetCount(&nGpu));
          if (iGpu < 0 || iGpu >= nGpu)
          {
              std::ostringstream err;
              err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
              throw std::invalid_argument(err.str());
          }
  
          CUcontext cuContext = NULL;
          createCudaContext(&cuContext, iGpu, 0);
  
          FFmpegDemuxer demuxer(szInFilePath.c_str());
          // Here set bLowLatency=true in the constructor.
          // Please don't use this flag except for low latency, it is harder to get 100% utilization of
          // hardware decoder with this flag set.
          NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false);
  
          int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0, n = 0;
          uint8_t *pVideo = NULL, **ppFrame;
          int64_t *pTimestamp;
          cv::Mat mat_bgr;
          std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};
          cv::namedWindow("a", cv::WINDOW_NORMAL);
  
          do
          {
              demuxer.Demux(&pVideo, &nVideoBytes);
              // Set flag CUVID_PKT_ENDOFPICTURE to signal that a complete packet has been sent to decode
              dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned, CUVID_PKT_ENDOFPICTURE, &pTimestamp, n++);
              if (!nFrame && nFrameReturned)
              {
                  LOG(INFO) << dec.GetVideoInfo();
                  LOG(INFO) << "Output format: " << aszDecodeOutFormat[dec.GetOutputFormat()];
              }
              if (nFrameReturned < 1)
                  continue;
              nFrame += nFrameReturned;
              cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 1.5, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
              cv::Mat mat_rgb = cv::Mat(dec.GetHeight(), dec.GetWidth(), CV_8UC3);
              cv::cvtColor(mat_yuv, mat_rgb, cv::COLOR_YUV2BGR_NV21);
              cv::cvtColor(mat_rgb, mat_bgr, cv::COLOR_RGB2BGR);
  
              cv::imshow("a", mat_bgr);
              char c = (char)cv::waitKey(25);
              if (c == 27)
                  break;
  
              // For a stream without B-frames, "one in and one out" is expected, and nFrameReturned should be always 1 for each input packet
              LOG(INFO) << "Decode: nVideoBytes=" <<   std::setw(10) << nVideoBytes
                        << ", nFrameReturned=" << std::setw  (10) << nFrameReturned
                        << ", total=" << std::setw(10) <<   nFrame;
              for (int i = 0; i < nFrameReturned; i++)
              {
                  LOG(INFO) << "Timestamp: " << pTimestamp  [i];
              }
  
          } while (nVideoBytes);
      }
      catch (const std::exception &ex)
      {
          LOG(ERROR) << ex.what();
          exit(1);
      }
      return 0;
  }
  ```
</details>

# <span style="color:red">♤ PYTHON BINDING ♠

# <span style="color:red">♡ BASIC ENCODING FLOW USING NVENCODE API ♥

The NVIDIA video encoder API is designed to accept raw video frames (in `YUV` or `RGB` 
format) and output the `H.264` or `HEVC` bitstream

## Step <a name="introduction"></a>
1. [Initialize the encoder.]
2. [Set up the desired encoding parameters.]
3. [Allocate input/output buffers.]
4. [Copy frames to input buffers and read bitstream from the output buffers. This can be done synchronously (Windows & Linux) or asynchronously (Windows 7 and above only).]
5. Clean-up - release all allocated input/output buffers.
6. [Close the encoding session.]

**_Note:_**

- `Bitstream`: data found in a stream of bits used in digital communication or data storage application.

- Before creating encoder, we should query capabilities of GPU to ensure that required functional is supported.

<details>
  <summary>Click to expand!</summary>

  ```cpp
  void ShowEncoderCapability()
  {
      ck(cuInit(0));
      int nGpu = 0;
      ck(cuDeviceGetCount(&nGpu));
      std::cout << "Encoder Capability" << std::endl
                << std::endl;
      for (int iGpu = 0; iGpu < nGpu; iGpu++)
      {
          CUdevice cuDevice = 0;
          ck(cuDeviceGet(&cuDevice, iGpu));
          char szDeviceName[80];
          ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
          CUcontext cuContext = NULL;
          ck(cuCtxCreate(&cuContext, 0, cuDevice));
          NvEncoderCuda enc(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);
  
          std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl
                    << std::endl;
          std::cout << "\tH264:\t\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl
                    << "\tH264_444:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl
                    << "\tH264_ME:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl
                    << "\tH264_WxH:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl
                    << "\tHEVC:\t\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl
                    << "\tHEVC_Main10:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no") << std::endl
                    << "\tHEVC_Lossless:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no") << std::endl
                    << "\tHEVC_SAO:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no") << std::endl
                    << "\tHEVC_444:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl
                    << "\tHEVC_ME:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl
                    << "\tHEVC_WxH:\t"
                    << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl;
  
          std::cout << std::endl;
  
          enc.DestroyEncoder();
          ck(cuCtxDestroy(cuContext));
      }
  }
  ```
</details>