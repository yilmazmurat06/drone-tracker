#pragma once
// ============================================================================
//  IStabilizer — P1: ego-motion (kendi hareketimizden kaynaklanan) bozulmayı
//  giderme arayüzü.
//
//  FİKİR: Kamera titreşen/dönen bir drone üzerinde. Arka plan, bizim
//  hareketimiz yüzünden kayar. Telemetri (gyro/attitude) + sparse optical flow
//  ile kameranın hareketini kestirip kareyi WARP ederek arka planı sabitleriz.
//  Geriye yalnızca HEDEFİN gerçek hareketi kalır → tespit kolaylaşır.
//
//  DURUM: Adım 1'de yalnızca BİLDİRİM (arayüz). İmplementasyonlar Adım 3'te,
//         örn. GyroFlowStabilizer.
// ============================================================================
#include <vector>
#include <opencv2/core.hpp>
#include "dtrack/core/types.hpp"

namespace dtrack {

class IStabilizer {
public:
    virtual ~IStabilizer() = default;

    // 'in' karesini, ona zamanca yakın telemetri örnekleriyle stabilize eder.
    //   telemetry_window : 'in'in t_rel'i etrafındaki örnekler (interpolasyon için)
    //   out              : warp edilmiş (sabitlenmiş) kare
    //   homography       : uygulanan 3x3 dönüşüm (sonraki aşamalar için)
    // return false → stabilize edilemedi (örn. takip edilecek köşe yok → saf gyro'ya düş)
    virtual bool stabilize(const Frame& in,
                           const std::vector<Telemetry>& telemetry_window,
                           Frame& out,
                           cv::Matx33f& homography) = 0;
};

} // namespace dtrack
