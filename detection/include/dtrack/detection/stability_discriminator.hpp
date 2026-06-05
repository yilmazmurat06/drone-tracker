#pragma once
//
// StabilityDiscriminator: zamansal kararlılık + geometrik özelliklerle
// drone/çöp ayrımı (IDiscriminator, Problem 3).
//
// Temel fikir: gerçek hedef her kare YAKIN konumda görünür; sahte tespit
// (gürültü, tek karelik parlama) rastgele konumlarda belirip kaybolur.
// Her yeni tespit için, geçmiş tespitler arasında uzamsal yakınlık eşlemesi
// yapar — eşleşen varsa yüksek skor, yoksa salt geometrik skor.
//
// Skor bileşenleri:
//   - Geometrik (anlık): alan + en/boy + parlaklık -> [0,1]
//   - Zamansal varlık : yakın geçmişte bu konumda tespit var mı? -> [0,1]
//   - Özellik kararlılığı: geçmişteki özelliklerle tutarlılık -> [0,1]
//   - Birleşik: 0.35*geometrik + 0.35*zamansal + 0.30*kararlilik

#include <cstdint>
#include <deque>
#include <vector>

#include <opencv2/core.hpp>

#include "dtrack/common/types.hpp"
#include "dtrack/detection/discriminator.hpp"

namespace dtrack::detection {

struct DiscriminatorConfig {
    // Uzamsal eşleme yarıçapı (piksel). Bu mesafe içinde geçmiş tespit varsa
    // zamansal skor yüksek olur.
    float match_radius = 25.0f;

    // Geçmiş tamponunun kare sayısı.
    int history_frames = 20;

    // Geometrik skor parametreleri.
    float area_min = 3.0f;
    float area_opt_min = 6.0f;
    float area_opt_max = 40.0f;
    float area_max = 60.0f;
    float aspect_sigma = 0.6f;
    float intensity_center = 80.0f;
    float intensity_sigma = 40.0f;

    float min_score = 0.15f;
};

class StabilityDiscriminator : public IDiscriminator {
public:
    explicit StabilityDiscriminator(DiscriminatorConfig cfg = {});

    void score(std::vector<common::Detection>& detections) override;
    void reset() override;

private:
    struct HistoryEntry {
        cv::Point2f pos;
        float area;
        float aspect;
        float intensity;
        int age{0};  // kaç kare önce eklendi
    };

    float geom_score(const common::Detection& d) const;

    DiscriminatorConfig cfg_;
    std::deque<HistoryEntry> history_;  // son N karenin tüm tespitleri
    int frame_count_{0};
};

}  // namespace dtrack::detection
