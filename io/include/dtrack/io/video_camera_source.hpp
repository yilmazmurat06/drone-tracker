#pragma once
//
// VideoCameraSource: cv::VideoCapture sarmalayıcı. SİMÜLATÖR ve gerçek donanım
// girişinin ana kapısı.
//
// Aynı sınıf şunların hepsini açar (cv::VideoCapture ne açıyorsa):
//   - Drone simülatörü video akışı:  "rtsp://...", "udp://@:5600", GStreamer boru hattı
//   - Kaydedilmiş klip:              "ucus.mp4"
//   - Gerçek donanım (ARM/Linux):    "/dev/video0" veya kamera indeksi 0
//
// Simülatör notları:
//   - AirSim:   görüntüyü RPC ile de alabilirsin; en hızlı entegrasyon, simülatörü
//               bir RTSP/UDP yayını verecek şekilde kurup buraya URL vermek.
//   - Gazebo:   gz-transport/ROS topic'i bir gst boru hattına köprüle, sonra aç.
//   - Genel:    simülatör ekranını/penceresini bir video dosyasına kaydedip
//               offline (realtime=false) de besleyebilirsin.
//
// Gri tonlama: pipeline tek kanal (CV_8UC1) bekler; renkli gelen kareyi çeviririz.

#include <string>

#include <opencv2/videoio.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/io/camera_source.hpp"

namespace dtrack::io {

class VideoCameraSource : public ICameraSource {
public:
    // uri: dosya yolu / akış URL'si / GStreamer boru hattı.
    // api: cv::CAP_ANY (otomatik), cv::CAP_GSTREAMER, cv::CAP_V4L2, cv::CAP_FFMPEG...
    VideoCameraSource(std::string uri, common::Modality modality,
                      int api_preference = cv::CAP_ANY);
    // Sayısal cihaz indeksiyle (örn. webcam 0).
    VideoCameraSource(int device_index, common::Modality modality,
                      int api_preference = cv::CAP_ANY);

    bool open() override;
    void close() override;
    bool is_open() const override { return cap_.isOpened(); }

    std::optional<common::FramePtr> next_frame() override;
    common::Modality modality() const override { return modality_; }

private:
    std::string uri_;
    int device_index_{-1};
    bool use_index_{false};
    common::Modality modality_;
    int api_pref_;

    cv::VideoCapture cap_;
    std::uint64_t frame_index_{0};
};

}  // namespace dtrack::io
