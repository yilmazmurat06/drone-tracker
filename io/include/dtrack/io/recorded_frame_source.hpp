#pragma once
// ============================================================================
//  RecordedFrameSource — kayıtlı .mp4 dosyasından kare okur.
//  (data/flight_XX.mp4 → OpenCV VideoCapture, 1920x1080 @ 60fps)
//
//  DURUM: Bu Adım 1'de yalnızca BİLDİRİM. Gerçek implementasyon Adım 2'de
//         io/src/recorded_frame_source.cpp içinde yazılacak.
//
//  ÖĞREN: "pimpl" (pointer-to-implementation) deseni — VideoCapture gibi ağır
//  OpenCV tiplerini header'dan gizler. Header'ı kullanan kod OpenCV'nin video
//  modülünü include etmek zorunda kalmaz; derleme bağımlılığı azalır.
// ============================================================================
#include <memory>
#include <string>
#include "dtrack/io/frame_source.hpp"

namespace dtrack {

class RecordedFrameSource : public IFrameSource {
public:
    // path: .mp4 dosya yolu. fps: kare zaman damgalarını üretmek için (vars. 60).
    explicit RecordedFrameSource(std::string path, double fps = 60.0);
    ~RecordedFrameSource() override;

    bool next(Frame& out) override;

    bool    is_open() const;      // video açılabildi mi?
    double  fps() const;          // kullanılan kare hızı (dosyadan veya parametreden)
    int64_t frame_count() const;  // toplam kare sayısı (biliniyorsa, yoksa 0)
    void    seek(int64_t frame_idx); // verilen kareye atla (next_id de güncellenir)

private:
    struct Impl;                      // tanımı .cpp'de
    std::unique_ptr<Impl> impl_;
};

} // namespace dtrack
