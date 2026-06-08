// ============================================================================
//  RecordedFrameSource implementasyonu — .mp4'ten kare okur (OpenCV VideoCapture).
//
//  ÖĞREN (pimpl gerçekleşti): Impl struct'ı burada tanımlıyoruz; cv::VideoCapture
//  yalnızca bu .cpp'de görünür, header temiz kalır.
//
//  Zaman damgası: kare i, video başından i/fps saniye sonradır. Telemetrideki
//  t_rel de "video başına göre saniye" olduğu için bu doğrudan eşleşir
//  (t_rel ≈ i/fps). Frame.t'yi ns olarak saklıyoruz.
// ============================================================================
#include "dtrack/io/recorded_frame_source.hpp"

#include <opencv2/videoio.hpp>

#include "dtrack/core/time.hpp"

namespace dtrack {

struct RecordedFrameSource::Impl {
    cv::VideoCapture cap;
    double  fps = 60.0;
    int64_t next_id = 0;
    int64_t count = 0;
};

RecordedFrameSource::RecordedFrameSource(std::string path, double fps)
    : impl_(std::make_unique<Impl>()) {
    impl_->cap.open(path);
    // Kare hızını dosyadan al; makul değilse parametreye düş.
    const double file_fps = impl_->cap.get(cv::CAP_PROP_FPS);
    impl_->fps = (file_fps > 1.0 && file_fps < 1000.0) ? file_fps : fps;
    impl_->count = static_cast<int64_t>(impl_->cap.get(cv::CAP_PROP_FRAME_COUNT));
}

RecordedFrameSource::~RecordedFrameSource() = default;

bool RecordedFrameSource::next(Frame& out) {
    if (!impl_->cap.isOpened()) return false;
    cv::Mat img;
    if (!impl_->cap.read(img) || img.empty()) return false;  // dosya sonu

    out.image = img;                 // not: cv::Mat referans-sayımlı; kopya ucuz
    out.id = impl_->next_id++;
    out.t  = seconds_to_ns(static_cast<double>(out.id) / impl_->fps);
    return true;
}

bool    RecordedFrameSource::is_open() const     { return impl_->cap.isOpened(); }
double  RecordedFrameSource::fps() const         { return impl_->fps; }
int64_t RecordedFrameSource::frame_count() const { return impl_->count; }

} // namespace dtrack
