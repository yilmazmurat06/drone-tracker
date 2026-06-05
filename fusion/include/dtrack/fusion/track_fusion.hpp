#pragma once
//
// ITrackFusion: çoklu kamera (görünür + termal) track-seviyesi füzyonunun
// soyut arayüzü (bkz. Problem 5).
//
// Tasarım kararı: GEÇ (late / track-seviyesi) füzyon. Her kamera kendi
// pipeline'ında bağımsız track üretir; burada YALNIZCA track seviyesinde
// birleştirilir. İki kamera aynı bölgede hemfikirse güven artar; tek kamera
// daha temkinli. Geometrik hizalama (FOV/optik merkez farkı) ve zaman eşleştirme
// burada ele alınır. Basit ağırlıklı oylama yeterli.

#include "dtrack/common/types.hpp"

#include <vector>

namespace dtrack::fusion {

class ITrackFusion {
public:
    virtual ~ITrackFusion() = default;

    // Farklı kameralardan gelen track kümelerini tek, birleşik track listesine
    // indirger. Çağrı başına bir kameranın güncellemesini besleyebilir veya
    // hepsini birden verebilirsin; implementasyon zaman penceresiyle eşleştirir.
    virtual std::vector<common::Track> fuse(
        const std::vector<common::Track>& visible_tracks,
        const std::vector<common::Track>& thermal_tracks) = 0;

    virtual void reset() = 0;
};

}  // namespace dtrack::fusion
