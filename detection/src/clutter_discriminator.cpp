// ============================================================================
//  ClutterDiscriminator implementasyonu — P3 klasik eleyici.
// ============================================================================
#include "dtrack/detection/clutter_discriminator.hpp"

#include <algorithm>
#include <cmath>

namespace dtrack {

float ClutterDiscriminator::score(const Detection& d) {
    const int bw = d.bbox.width;
    const int bh = d.bbox.height;
    if (bw <= 0 || bh <= 0) return 0.f;

    // 1. Doluluk: gerçek piksel alanı / bounding-box alanı → [0,1].
    //    Drone gibi kompakt cisim ≈1; ince streak veya L-şekilli artefakt < 0.5.
    const float box_area = static_cast<float>(bw * bh);
    const float compact  = std::min(1.f, d.area / box_area);

    // 2. En-boy oranı: log ölçeğinde Gauss → AR=1 için skor=1, uzadıkça düşer.
    //    Hem AR>1 hem AR<1 (dikey streak) simetrik ceza alır.
    const float ar     = d.aspect_ratio > 0.f ? d.aspect_ratio : 1.f;
    const float ar_log = std::log(ar);
    const float ar_sc  = std::exp(-(ar_log * ar_log) /
                                  (2.f * p_.ar_sigma * p_.ar_sigma));

    // 3. Alan bandı: [area_min, area_max] arasında 1.0; dışında yumuşak ceza.
    float area_sc = 1.f;
    if (d.area < p_.area_min && p_.area_min > 0.f) {
        area_sc = d.area / p_.area_min;
    } else if (d.area > p_.area_max && p_.area_max > 0.f) {
        area_sc = p_.area_max / d.area;  // büyüdükçe küçülür
    }
    area_sc = std::max(0.f, area_sc);

    return p_.w_compact * compact + p_.w_ar * ar_sc + p_.w_area * area_sc;
}

} // namespace dtrack
