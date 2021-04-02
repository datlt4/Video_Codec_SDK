#include "main.hpp"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main()
{
#ifdef NVIDIA_DECODER
    std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";
    // std::string szInFilePath = "rtsp://admin:esbt1234@10.236.1.105:554/onvif1";
    // std::string szInFilePath = "rtsp://0.0.0.0:8554/vlc";

#ifdef APP_DEC_CPP
    ShowDecoderCapability();
    AppDec(szInFilePath);
#endif //APP_DEC_CPP

#ifdef APP_DEC_LOW_LATENCY_CPP
    ShowDecoderCapability();
    AppDecLowLatency(szInFilePath);
#endif //APP_DEC_LOW_LATENCY_CPP

#ifdef APP_IN_THREAD_CPP
    VideoStreamer *streamer = new VideoStreamer((char *)szInFilePath.c_str(), 0);
    std::thread *thread = new std::thread(&VideoStreamer::update, streamer);
    bool state;
    cv::Mat mat_bgr;
    uchar *str;
    cv::namedWindow("a", cv::WINDOW_AUTOSIZE);
    while (true)
    {
        if (streamer->status())
        {
            streamer->read(state, mat_bgr);
            if (state)
            {
                cv::imshow("a", mat_bgr);
                char c = (char)cv::waitKey(1000 / 30.);
                if (c == 27)
                    break;
            }
        }
    }
    thread->detach();
#endif //APP_IN_THREAD_CPP
#endif //NVIDIA_DECODER

#ifdef NVIDIA_ENCODER
#ifdef APP_ENC_CUDA_CPP
    AppEncCuda();
#endif //APP_ENC_CUDA_CPP
#ifdef APP_ENC_DEC_CPP
    AppEncDec();
#endif //APP_ENC_DEC_CPP
#endif //NVIDIA_ENCODER
    return 0;
    return 0;
}

void AppDec(std::string szInFilePath)
{
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
    cv::namedWindow("a", cv::WINDOW_AUTOSIZE);
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
        cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 3 / 2, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
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
}

void AppDecLowLatency(std::string szInFilePath)
{
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
            cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 3 / 2, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
            cv::Mat mat_rgb = cv::Mat(dec.GetHeight(), dec.GetWidth(), CV_8UC3);
            cv::cvtColor(mat_yuv, mat_rgb, cv::COLOR_YUV2BGR_NV21);
            cv::cvtColor(mat_rgb, mat_bgr, cv::COLOR_RGB2BGR);

            cv::imshow("a", mat_bgr);
            char c = (char)cv::waitKey(25);
            if (c == 27)
                break;

            // For a stream without B-frames, "one in and one out" is expected, and nFrameReturned should be always 1 for each input packet
            LOG(INFO) << "Decode: nVideoBytes=" << std::setw(10) << nVideoBytes
                      << ", nFrameReturned=" << std::setw(10) << nFrameReturned
                      << ", total=" << std::setw(10) << nFrame;
            for (int i = 0; i < nFrameReturned; i++)
            {
                LOG(INFO) << "Timestamp: " << pTimestamp[i];
            }

        } while (nVideoBytes);
        ck(cuCtxDestroy(cuContext));
        LOG(INFO) << "End of process";
    }
    catch (const std::exception &ex)
    {
        LOG(ERROR) << ex.what();
        exit(1);
    }
}

VideoStreamer::VideoStreamer(char *srcStr, int iGpu)
{
    ShowDecoderCapability();
    this->srcStr = new std::string(srcStr);
    this->iGpu = iGpu;
}

void VideoStreamer::update()
{
    try
    {
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (this->iGpu < 0 || this->iGpu >= nGpu)
        {
            std::ostringstream err;
            err << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            throw std::invalid_argument(err.str());
        }

        CUcontext cuContext = NULL;
        createCudaContext(&cuContext, this->iGpu, 0);
        FFmpegDemuxer demuxer(this->srcStr->c_str());
        // Here set bLowLatency=true in the constructor.
        // Please don't use this flag except for low latency, it is harder to get 100% utilization of
        // hardware decoder with this flag set.
        NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false);

        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0, n = 0;
        uint8_t *pVideo = NULL, **ppFrame;
        int64_t *pTimestamp;
        std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};
        // cv::namedWindow("a", cv::WINDOW_NORMAL);
        do
        {
            demuxer.Demux(&pVideo, &nVideoBytes);
            // Set flag CUVID_PKT_ENDOFPICTURE to signal that a complete packet has been sent to decode
            dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned, CUVID_PKT_ENDOFPICTURE, &pTimestamp, n++);
            if (!nFrame && nFrameReturned)
            {
                LOG(INFO) << dec.GetVideoInfo();
                LOG(INFO) << "Output format: " << aszDecodeOutFormat[dec.GetOutputFormat()];
                this->frame_height = dec.GetHeight();
                this->frame_width = dec.GetWidth();
            }

            if (nFrameReturned < 1)
                continue;
            nFrame += nFrameReturned;

            cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 3 / 2, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
            cv::Mat mat_rgb = cv::Mat(dec.GetHeight(), dec.GetWidth(), CV_8UC3);
            cv::cvtColor(mat_yuv, mat_rgb, cv::COLOR_YUV2BGR_NV21);
            cv::cvtColor(mat_rgb, this->mat_bgr, cv::COLOR_RGB2BGR);
            this->_status = true;

        } while (nVideoBytes);
        ck(cuCtxDestroy(cuContext));
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        exit(1);
    }
}

uchar *VideoStreamer::read()
{
    cv::Mat image;
    this->mat_bgr.copyTo(image);
    uchar *data = image.data;
    return data;
}

void VideoStreamer::read(bool &s, cv::Mat &image)
{
    s = this->_status;
    this->mat_bgr.copyTo(image);
}

bool VideoStreamer::status()
{
    return this->_status;
}

int VideoStreamer::getWidth()
{
    return this->frame_width;
}

int VideoStreamer::getHeight()
{
    return this->frame_height;
}

VideoStreamer::~VideoStreamer()
{
}

void ShowEncoderCapability()
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    LOG(INFO) << "Encoder Capability" << std::endl;
    for (int iGpu = 0; iGpu < nGpu; iGpu++)
    {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);

        LOG(INFO) << "GPU " << iGpu << " - " << szDeviceName << std::endl
                  << std::endl;
        LOG(INFO) << "H264:\t\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no");
        LOG(INFO) << "H264_444:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no");
        LOG(INFO) << "H264_ME:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no");
        LOG(INFO) << "H264_WxH:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX));
        LOG(INFO) << "HEVC:\t\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no");
        LOG(INFO) << "HEVC_Main10:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no");
        LOG(INFO) << "HEVC_Lossless:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no");
        LOG(INFO) << "HEVC_SAO:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no");
        LOG(INFO) << "HEVC_444:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no");
        LOG(INFO) << "HEVC_ME:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no");
        LOG(INFO) << "HEVC_WxH:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl;

        enc.DestroyEncoder();
        ck(cuCtxDestroy(cuContext));
    }
}

NvCUStream::NvCUStream(CUcontext cuDevice, int cuStreamType, std::unique_ptr<NvEncoderOutputInVidMemCuda> &pEnc)
{
    this->device = cuDevice;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(this->device));
    // Create CUDA streams
    if (cuStreamType == 1)
    {
        ck(cuStreamCreate(&this->inputStream, CU_STREAM_DEFAULT));
        this->outputStream = this->inputStream;
    }
    else if (cuStreamType == 2)
    {
        ck(cuStreamCreate(&this->inputStream, CU_STREAM_DEFAULT));
        ck(cuStreamCreate(&this->outputStream, CU_STREAM_DEFAULT));
    }
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));
    // Set input and output CUDA streams in driver
    pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR) & this->inputStream, (NV_ENC_CUSTREAM_PTR) & this->outputStream);
}

NvCUStream::~NvCUStream()
{
    ck(cuCtxPushCurrent(this->device));
    if (this->inputStream != NULL)
        cuStreamDestroy(this->inputStream);
    if (this->outputStream != NULL)
        cuStreamDestroy(this->outputStream);
    ck(cuCtxPopCurrent(NULL));
}

CUstream NvCUStream::GetOutputCUStream()
{
    return this->outputStream;
}

CUstream NvCUStream::GetInputCUStream()
{
    return this->inputStream;
}

CRC::CRC(CUcontext cuDevice, uint32_t bufferSize)
{
    this->device = cuDevice;
    ck(cuCtxPushCurrent(this->device));
    // Allocate video memory buffer to store CRC of encoded frame
    ck(cuMemAlloc(&this->crcVidMem, bufferSize));
    ck(cuCtxPopCurrent(NULL));
}

CRC::~CRC()
{
    ck(cuCtxPushCurrent(this->device));
    ck(cuMemFree(this->crcVidMem));
    ck(cuCtxPopCurrent(NULL));
}

void CRC::GetCRC(NV_ENC_OUTPUT_PTR pVideoMemBfr, CUstream outputStream)
{
    ComputeCRC((uint8_t *)pVideoMemBfr, (uint32_t *)this->crcVidMem, outputStream);
}

CUdeviceptr CRC::GetCRCVidMemPtr()
{
    return this->crcVidMem;
}

DumpVidMemOutput::DumpVidMemOutput(CUcontext cuDevice, uint32_t size, char *outFilePath, bool bUseCUStream)
{
    this->device = cuDevice;
    this->bfrSize = size;
    bCRC = bUseCUStream;

    ck(cuCtxPushCurrent(this->device));
    // Allocate host memory buffer to copy encoded output and CRC
    ck(cuMemAllocHost((void **)&this->pHostMemEncOp, (this->bfrSize + (this->bCRC ? 4 : 0))));
    ck(cuCtxPopCurrent(NULL));
    // Open file to dump CRC
    if (this->bCRC)
    {
        this->crcFile = std::string(outFilePath) + "_crc.txt";
        this->fpCRCOut.open(this->crcFile, std::ios::out);
        this->pHostMemCRC = (uint32_t *)((uint8_t *)this->pHostMemEncOp + this->bfrSize);
    }
}

DumpVidMemOutput::~DumpVidMemOutput()
{
    ck(cuCtxPushCurrent(this->device));
    ck(cuMemFreeHost(this->pHostMemEncOp));
    ck(cuCtxPopCurrent(NULL));
    if (this->bCRC)
    {
        this->fpCRCOut.close();
        LOG(INFO) << "CRC saved in file: " << this->crcFile;
    }
}

void DumpVidMemOutput::DumpOutputToFile(CUdeviceptr pEncFrameBfr, CUdeviceptr pCRCBfr, std::ofstream &fpOut, uint32_t nFrame)
{
    ck(cuCtxPushCurrent(this->device));
    // Copy encoded frame from video memory buffer to host memory buffer
    ck(cuMemcpyDtoH(this->pHostMemEncOp, pEncFrameBfr, this->bfrSize));
    // Copy encoded frame CRC from video memory buffer to host memory buffer
    if (this->bCRC)
    {
        ck(cuMemcpyDtoH(this->pHostMemCRC, pCRCBfr, 4));
    }
    ck(cuCtxPopCurrent(NULL));
    // Write encoded bitstream in file
    uint32_t offset = sizeof(NV_ENC_ENCODE_OUT_PARAMS);
    uint32_t bitstream_size = ((NV_ENC_ENCODE_OUT_PARAMS *)this->pHostMemEncOp)->bitstreamSizeInBytes;
    uint8_t *ptr = this->pHostMemEncOp + offset;
    fpOut.write((const char *)ptr, bitstream_size);
    // Write CRC in file
    if (bCRC)
    {
        if (!nFrame)
        {
            this->fpCRCOut << "Frame num" << std::setw(10) << "CRC" << std::endl;
        }
        this->fpCRCOut << std::dec << std::setfill(' ') << std::setw(5) << nFrame << "          ";
        this->fpCRCOut << std::hex << std::setfill('0') << std::setw(8) << *this->pHostMemCRC << std::endl;
    }
}

template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
    pEnc->CreateEncoder(&initializeParams);
}

void EncodeCuda(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut)
{
    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat));
    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);
    int nFrameSize = pEnc->GetFrameSize();
    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;
    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<std::vector<uint8_t>> vPacket;
        if (nRead == nFrameSize)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_HOST,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes);

            pEnc->EncodeFrame(vPacket);
        }
        else
        {
            pEnc->EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            // For each encoded packet
            fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
        }

        if (nRead != nFrameSize)
            break;
    }

    pEnc->DestroyEncoder();
    LOG(INFO) << "Total frames encoded: " << nFrame;
}

void EncodeCudaOpInVidMem(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut, char *outFilePath, int32_t cuStreamType)
{
    std::unique_ptr<NvEncoderOutputInVidMemCuda> pEnc(new NvEncoderOutputInVidMemCuda(cuContext, nWidth, nHeight, eFormat));
    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);
    int nFrameSize = pEnc->GetFrameSize();
    bool bUseCUStream = cuStreamType != -1 ? true : false;
    std::unique_ptr<CRC> pCRC;
    std::unique_ptr<NvCUStream> pCUStream;
    if (bUseCUStream)
    {
        // Allocate CUDA streams
        pCUStream.reset(new NvCUStream(reinterpret_cast<CUcontext>(pEnc->GetDevice()), cuStreamType, pEnc));
        // When CUDA streams are used, the encoded frame's CRC is computed using cuda kernel
        pCRC.reset(new CRC(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize()));
    }
    // For dumping output - encoded frame and CRC, to a file
    std::unique_ptr<DumpVidMemOutput> pDumpVidMemOutput(new DumpVidMemOutput(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize(), outFilePath, bUseCUStream));
    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;
    // Encoding loop
    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<NV_ENC_OUTPUT_PTR> pVideoMemBfr;
        if (nRead == nFrameSize)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_HOST,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes,
                                             false,
                                             bUseCUStream ? pCUStream->GetInputCUStream() : NULL);
            pEnc->EncodeFrame(pVideoMemBfr);
        }
        else
        {
            pEnc->EndEncode(pVideoMemBfr);
        }

        for (uint32_t i = 0; i < pVideoMemBfr.size(); ++i)
        {
            if (bUseCUStream)
            {
                // Compute CRC of encoded stream
                pCRC->GetCRC(pVideoMemBfr[0], pCUStream->GetOutputCUStream());
            }
            pDumpVidMemOutput->DumpOutputToFile((CUdeviceptr)(pVideoMemBfr[0]), bUseCUStream ? pCRC->GetCRCVidMemPtr() : 0, fpOut, nFrame);
            nFrame++;
        }
        if (nRead != nFrameSize)
            break;
    }
    pEnc->DestroyEncoder();
    LOG(INFO) << "Total frames encoded: " << nFrame;
}

void EncodeProc(CUdevice cuDevice, int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam *pEncodeCLIOptions, bool bBgra64, const char *szInFilePath, const char *szMediaPath, std::exception_ptr &encExceptionPtr)
{
    CUdeviceptr dpFrame = 0, dpBgraFrame = 0;
    CUcontext cuContext = NULL;

    try
    {
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, nWidth, nHeight, eFormat);
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        enc.CreateDefaultEncoderParams(&initializeParams, pEncodeCLIOptions->GetEncodeGUID(), pEncodeCLIOptions->GetPresetGUID());

        pEncodeCLIOptions->SetInitParams(&initializeParams, eFormat);

        enc.CreateEncoder(&initializeParams);

        std::ifstream fpIn(szInFilePath, std::ifstream::in | std::ifstream::binary);
        if (!fpIn)
        {
            std::cout << "Unable to open input file: " << szInFilePath << std::endl;
            return;
        }

        int nHostFrameSize = bBgra64 ? nWidth * nHeight * 8 : enc.GetFrameSize();
        std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nHostFrameSize]);
        CUdeviceptr dpBgraFrame = 0;
        ck(cuMemAlloc(&dpBgraFrame, nWidth * nHeight * 8));
        int nFrame = 0;
        std::streamsize nRead = 0;
        FFmpegStreamer streamer(pEncodeCLIOptions->IsCodecH264() ? AV_CODEC_ID_H264 : AV_CODEC_ID_HEVC, nWidth, nHeight, 25, szMediaPath);
        do
        {
            std::vector<std::vector<uint8_t>> vPacket;
            nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nHostFrameSize).gcount();
            if (nRead == nHostFrameSize)
            {
                const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();

                if (bBgra64)
                {
                    // Color space conversion
                    ck(cuMemcpyHtoD(dpBgraFrame, pHostFrame.get(), nHostFrameSize));
                    Bgra64ToP016((uint8_t *)dpBgraFrame, nWidth * 8, (uint8_t *)encoderInputFrame->inputPtr, encoderInputFrame->pitch, nWidth, nHeight);
                }
                else
                {
                    NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                                     (int)encoderInputFrame->pitch,
                                                     enc.GetEncodeWidth(),
                                                     enc.GetEncodeHeight(),
                                                     CU_MEMORYTYPE_HOST,
                                                     encoderInputFrame->bufferFormat,
                                                     encoderInputFrame->chromaOffsets,
                                                     encoderInputFrame->numChromaPlanes);
                }
                enc.EncodeFrame(vPacket);
            }
            else
            {
                enc.EndEncode(vPacket);
            }
            for (std::vector<uint8_t> &packet : vPacket)
            {
                streamer.Stream(packet.data(), (int)packet.size(), nFrame++);
            }
        } while (nRead == nHostFrameSize);
        ck(cuMemFree(dpBgraFrame));
        dpBgraFrame = 0;

        enc.DestroyEncoder();
        fpIn.close();

        std::cout << std::flush << "Total frames encoded: " << nFrame << std::endl
                  << std::flush;
    }
    catch (const std::exception &)
    {
        encExceptionPtr = std::current_exception();
        ck(cuMemFree(dpBgraFrame));
        dpBgraFrame = 0;
        ck(cuMemFree(dpFrame));
        dpFrame = 0;
    }
}

void DecodeProc(CUdevice cuDevice, const char *szMediaUri, OutputFormat eOutputFormat, const char *szOutFilePath, std::exception_ptr &decExceptionPtr)
{
    CUdeviceptr dpRgbFrame = 0;
    try
    {
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        FFmpegDemuxer demuxer(szMediaUri);
        // Output host frame for native format; otherwise output device frame for CUDA processing
        NvDecoder dec(cuContext, eOutputFormat != native, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, true);

        uint8_t *pVideo = NULL;
        int nVideoBytes = 0;
        int nFrame = 0;
        std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
        if (!fpOut)
        {
            std::ostringstream err;
            err << "Unable to open output file: " << szOutFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }

        const char *szTail = "\xe0\x00\x00\x00\x01\xce\x8c\x4d\x9d\x10\x8e\x25\xe9\xfe";
        int nWidth = demuxer.GetWidth(), nHeight = demuxer.GetHeight();
        std::unique_ptr<uint8_t[]> pRgbFrame;
        int nRgbFramePitch = 0, nRgbFrameSize = 0;
        if (eOutputFormat != native)
        {
            nRgbFramePitch = nWidth * (eOutputFormat == bgra ? 4 : 8);
            nRgbFrameSize = nRgbFramePitch * nHeight;
            pRgbFrame.reset(new uint8_t[nRgbFrameSize]);
            ck(cuMemAlloc(&dpRgbFrame, nRgbFrameSize));
        }
        do
        {
            demuxer.Demux(&pVideo, &nVideoBytes);
            uint8_t **ppFrame;
            int nFrameReturned = 0;
            dec.Decode(nVideoBytes > 0 ? pVideo + 6 : NULL,
                       // Cut head and tail generated by FFmpegDemuxer
                       nVideoBytes - (nVideoBytes > 20 && !memcmp(pVideo + nVideoBytes - 14, szTail, 14) ? 20 : 6),
                       &ppFrame, &nFrameReturned, CUVID_PKT_ENDOFPICTURE);
            int iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
            if (!nFrame)
            {
                LOG(INFO) << "Color matrix coefficient: " << iMatrix;
            }
            for (int i = 0; i < nFrameReturned; i++)
            {
                if (eOutputFormat == native)
                {
                    fpOut.write(reinterpret_cast<char *>(ppFrame[i]), dec.GetFrameSize());
                }
                else
                {
                    // Color space conversion
                    if (dec.GetBitDepth() == 8)
                    {
                        if (eOutputFormat == bgra)
                        {
                            Nv12ToColor32<BGRA32>(ppFrame[i], nWidth, (uint8_t *)dpRgbFrame, nRgbFramePitch, nWidth, nHeight, iMatrix);
                        }
                        else
                        {
                            Nv12ToColor64<BGRA64>(ppFrame[i], nWidth, (uint8_t *)dpRgbFrame, nRgbFramePitch, nWidth, nHeight, iMatrix);
                        }
                    }
                    else
                    {
                        if (eOutputFormat == bgra)
                        {
                            P016ToColor32<BGRA32>(ppFrame[i], nWidth * 2, (uint8_t *)dpRgbFrame, nRgbFramePitch, nWidth, nHeight, iMatrix);
                        }
                        else
                        {
                            P016ToColor64<BGRA64>(ppFrame[i], nWidth * 2, (uint8_t *)dpRgbFrame, nRgbFramePitch, nWidth, nHeight, iMatrix);
                        }
                    }
                    ck(cuMemcpyDtoH(pRgbFrame.get(), dpRgbFrame, nRgbFrameSize));
                    fpOut.write(reinterpret_cast<char *>(pRgbFrame.get()), nRgbFrameSize);
                }
                nFrame++;
            }
        } while (nVideoBytes);
        if (eOutputFormat != native)
        {
            ck(cuMemFree(dpRgbFrame));
            dpRgbFrame = 0;
            pRgbFrame.reset(nullptr);
        }
        fpOut.close();

        std::cout << "Total frame decoded: " << nFrame << std::endl
                  << "Saved in file " << szOutFilePath << " in "
                  << (eOutputFormat == native ? (dec.GetBitDepth() == 8 ? "nv12" : "p010") : (eOutputFormat == bgra ? "bgra" : "bgra64"))
                  << " format" << std::endl;
    }
    catch (const std::exception &)
    {
        decExceptionPtr = std::current_exception();
        cuMemFree(dpRgbFrame);
    }
}

void AppEncCuda()
{
    std::string szInFilePath = "/home/m/Documents/NVCODEC/HeavyHand_1080p.yuv";
    std::string szOutFilePath = "HeavyHand_1080p.h264";
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    NvEncoderInitParam encodeCLIOptions("");
    int cuStreamType = -1;
    bool bOutputInVideoMem = false;
    int iGpu = 0;
    ShowEncoderCapability();
    try
    {
        CheckInputFile(szInFilePath.c_str());
        ValidateResolution(nWidth, nHeight);

        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu)
        {
            LOG(ERROR) << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
            return;
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        LOG(INFO) << "GPU in use: " << szDeviceName;
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        // Open input file
        std::ifstream fpIn(szInFilePath, std::ifstream::in | std::ifstream::binary);
        if (!fpIn)
        {
            std::ostringstream err;
            err << "Unable to open input file: " << szInFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }

        // Open output file
        std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
        if (!fpOut)
        {
            std::ostringstream err;
            err << "Unable to open output file: " << szOutFilePath << std::endl;
            throw std::invalid_argument(err.str());
        }

        // Encode
        if (bOutputInVideoMem)
        {
            EncodeCudaOpInVidMem(nWidth, nHeight, eFormat, encodeCLIOptions, cuContext, fpIn, fpOut, (char *)szOutFilePath.c_str(), cuStreamType);
        }
        else
        {
            EncodeCuda(nWidth, nHeight, eFormat, encodeCLIOptions, cuContext, fpIn, fpOut);
        }

        fpOut.close();
        fpIn.close();

        LOG(INFO) << "Bitstream saved in file " << szOutFilePath;
    }
    catch (const std::exception &e)
    {
        LOG(ERROR) << e.what();
    }
}

void AppEncDec()
{
    std::string szInFilePath = "/home/m/Documents/NVCODEC/HeavyHand_1080p.yuv";
    std::string szOutFilePath = "HeavyHand_1080p.h264";
    int nWidth = 1920, nHeight = 1080;
    NV_ENC_BUFFER_FORMAT eInputFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    OutputFormat eOutputFormat = native;
    NvEncoderInitParam encodeCLIOptions("", NULL);
    int iGpu = 0;
    bool bBgra64 = false;
    std::exception_ptr encExceptionPtr;
    std::exception_ptr decExceptionPtr;
    try
    {
        CheckInputFile(szInFilePath.c_str());
        ValidateResolution(nWidth, nHeight);

        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu)
        {
            LOG(INFO) << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]";
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        LOG(INFO) << "GPU in use: " << szDeviceName;

        const char *szMediaUri = "tcp://127.0.0.1:8899";
        char szMediaUriDecode[1024];
        sprintf(szMediaUriDecode, "%s?listen", szMediaUri);
        LOG(INFO) << szMediaUriDecode;
        // std::thread *thDecode = new std::thread(DecodeProc, cuDevice, szMediaUriDecode, eOutputFormat, szOutFilePath, std::ref(decExceptionPtr));
        // std::thread *thEncode = new std::thread(EncodeProc, cuDevice, nWidth, nHeight, eInputFormat, &encodeCLIOptions, bBgra64, szInFilePath, szMediaUri, std::ref(encExceptionPtr));
        // thEncode->join();
        // thDecode->join();

        if (encExceptionPtr)
        {
            std::rethrow_exception(encExceptionPtr);
        }
        if (decExceptionPtr)
        {
            std::rethrow_exception(decExceptionPtr);
        }
    }
    catch (const std::exception &e)
    {
        LOG(ERROR) << e.what();
    }
}
