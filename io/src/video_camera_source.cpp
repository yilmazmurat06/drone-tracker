#include "dtrack/io/video_camera_source.hpp"

#include <chrono>
#include <string>

#include <opencv2/imgproc.hpp>

#include "dtrack/common/time.hpp"

namespace dtrack::io {

VideoCameraSource::VideoCameraSource(std::string uri, common::Modality modality,
                                     int api_preference)
    : uri_(std::move(uri)), use_index_(false), modality_(modality), api_pref_(api_preference) {}

VideoCameraSource::VideoCameraSource(int device_index, common::Modality modality,
                                     int api_preference)
    : device_index_(device_index), use_index_(true), modality_(modality),
      api_pref_(api_preference) {}

namespace {
// Akış URL'si mi? (rtsp/udp/http/tcp/rtp -> canlı, duvar-saati kullan).
bool is_stream_uri(const std::string& u) {
    auto starts = [&](const char* p) { return u.rfind(p, 0) == 0; };
    return starts("rtsp://") || starts("udp://") || starts("rtp://") ||
           starts("http://") || starts("https://") || starts("tcp://") ||
           u.rfind("gst", 0) == 0 || u.find("appsink") != std::string::npos;
}
}  // namespace

bool VideoCameraSource::open() {
    frame_index_ = 0;
    bool ok;
    if (use_index_) {
        ok = cap_.open(device_index_, api_pref_);
    } else {
        ok = cap_.open(uri_, api_pref_);
    }
    if (!ok) return false;

    // Çevrimdışı DOSYA ise zaman damgasını videonun kare hızından türet
    // (kare_no / fps). Canlı cihaz/akışta gerçek-zaman (duvar-saati) doğru.
    t0_ = common::now();
    media_fps_ = cap_.get(cv::CAP_PROP_FPS);
    const bool is_file = !use_index_ && !is_stream_uri(uri_);
    use_media_time_ = is_file && media_fps_ > 1.0 && media_fps_ < 1000.0;
    return true;
}

void VideoCameraSource::close() { cap_.release(); }

std::optional<common::FramePtr> VideoCameraSource::next_frame() {
    if (!cap_.isOpened()) return std::nullopt;

    cv::Mat raw;
    if (!cap_.read(raw) || raw.empty()) {
        return std::nullopt;  // akış bitti / kare henüz hazır değil
    }

    auto frame = std::make_shared<common::Frame>();
    frame->index = frame_index_;
    // Dosya: damga = t0 + kare_no/fps -> işlem hızından bağımsız, düzgün dt.
    // Canlı cihaz/akış: damga = okuma anı (duvar-saati) -> gerçek varış zamanı.
    if (use_media_time_) {
        const double sec = static_cast<double>(frame_index_) / media_fps_;
        frame->stamp = t0_ + std::chrono::duration_cast<common::Duration>(
                                 std::chrono::duration<double>(sec));
    } else {
        frame->stamp = common::now();
    }
    ++frame_index_;
    frame->modality = modality_;

    // Pipeline tek kanal bekler: renkliyse gri tonlamaya çevir.
    if (raw.channels() == 3) {
        cv::cvtColor(raw, frame->image, cv::COLOR_BGR2GRAY);
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, frame->image, cv::COLOR_BGRA2GRAY);
    } else {
        frame->image = raw.clone();  // kaynak tamponu yeniden kullanır -> kopyala
    }

    return common::FramePtr(frame);
}

}  // namespace dtrack::io
