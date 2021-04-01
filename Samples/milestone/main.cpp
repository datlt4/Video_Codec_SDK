#include "main.hpp"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main()
{
    ShowDecoderCapability();
    std::string szInFilePath = "/home/m/Documents/NVCODEC/your_name.mp4";
    // std::string szInFilePath = "rtsp://admin:esbt1234@10.236.1.105:554/onvif1";
    // std::string szInFilePath = "rtsp://0.0.0.0:8554/vlc";
    // CheckInputFile(szInFilePath.c_str());

    // AppDec(szInFilePath);
    AppDecLowLatency(szInFilePath);

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
            cv::Mat mat_yuv = cv::Mat(dec.GetHeight() * 1.5, dec.GetWidth(), CV_8UC1, ppFrame[nFrameReturned - 1]);
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
    }
    catch (const std::exception &ex)
    {
        LOG(ERROR) << ex.what();
        exit(1);
    }
}
