#include "dtrack/io/video_camera_source.hpp"

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

bool VideoCameraSource::open() {
    frame_index_ = 0;
    if (use_index_) {
        return cap_.open(device_index_, api_pref_);
    }
    return cap_.open(uri_, api_pref_);
}

void VideoCameraSource::close() { cap_.release(); }

std::optional<common::FramePtr> VideoCameraSource::next_frame() {
    if (!cap_.isOpened()) return std::nullopt;

    cv::Mat raw;
    if (!cap_.read(raw) || raw.empty()) {
        return std::nullopt;  // akış bitti / kare henüz hazır değil
    }

    auto frame = std::make_shared<common::Frame>();
    frame->index = frame_index_++;
    // Zaman damgasını okuma anına ata. (Donanım zaman damgası varsa ileride
    // CAP_PROP_POS_MSEC ya da sürücüden alınıp burada kullanılabilir.)
    frame->stamp = common::now();
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
