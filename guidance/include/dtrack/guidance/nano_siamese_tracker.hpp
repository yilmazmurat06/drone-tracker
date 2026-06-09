#pragma once
// ============================================================================
//  NanoSiameseTracker — FAZ 1 SPIKE: hazır (eğitimsiz) hafif Siamese tracker.
//
//  AMAÇ (öğrenerek ilerleme / risk azaltma): Kendi int8 ağımızı eğitmeden ÖNCE,
//  jenerik bir Siamese tracker'ın bizim hava-hava (gök önünde küçük drone)
//  videolarımızda NCC stub'tan iyi takip edip etmediğini UCUZA ölçmek. Sinyal
//  olumluysa kendi ağımızı eğitmeye değer; hiç tutmuyorsa sorun model küçüklüğü
//  değil domain/çözünürlüktür → eğitim de çözmez.
//
//  ALTYAPI: OpenCV'nin yerleşik `cv::TrackerNano`'su (NanoTrack v2) — ~1.9MB,
//  iki ONNX (backbone + neckhead). Bizim 2.5MB bütçemize boyut olarak yakın bir
//  ANALOG (ama bu spike CPU/DNN üzerinde koşar, NPU değil — amaç doğruluk değil
//  sinyaldir). `getTrackingScore()` → confidence = bizim "precision" metriğimiz.
//
//  Aynı `ISingleTargetTracker` sözleşmesini uygular → GuidanceController hiç
//  değişmeden NccTemplateTracker ile takas edilebilir. Bu, soyutlamamızın
//  ikinci somut testidir.
// ============================================================================
#include <string>

#include <opencv2/video/tracking.hpp>   // cv::TrackerNano

#include "dtrack/guidance/single_target_tracker.hpp"

namespace dtrack {

class NanoSiameseTracker : public ISingleTargetTracker {
public:
    struct Params {
        std::string backbone = "models/nanotrack_backbone_sim.onnx";
        std::string neckhead = "models/nanotrack_head_sim.onnx";
    };

    NanoSiameseTracker() : NanoSiameseTracker(Params{}) {}
    explicit NanoSiameseTracker(Params p);

    void init(const cv::Mat& frame, const cv::Rect& bbox) override;
    STResult track(const cv::Mat& frame) override;

private:
    Params               p_;
    cv::Ptr<cv::TrackerNano> trk_;   // her init()'te yeniden kurulur
    cv::Rect             last_box_;
    bool                 ready_ = false;
};

} // namespace dtrack
