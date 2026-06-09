// ============================================================================
//  MovingTargetDetector implementasyonu — hibrit (kare-farkı + MOG2).
// ============================================================================
#include "dtrack/detection/moving_target_detector.hpp"

#include <functional>
#include <map>
#include <vector>

#include <opencv2/imgproc.hpp>   // cvtColor, threshold, morphology, connectedComponents

namespace dtrack {

namespace {
cv::Mat to_gray(const cv::Mat& img) {
    if (img.channels() == 1) return img;
    cv::Mat g;
    cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
    return g;
}

// Birbirine 'gap' pikselden yakın Detection'ları tek hedefe birleştirir.
// Union-find: kutuları 'gap' kadar şişirip kesişenleri aynı gruba bağla; her
// grubu birleşik-bbox + alan-toplamı + alan-ağırlıklı centroid ile tek aday yap.
// AMAÇ: büyük cismin (zeplin) parça parça blob'larını tek ize toplamak.
void merge_detections(std::vector<Detection>& dets, int gap) {
    const int n = static_cast<int>(dets.size());
    if (n < 2) return;

    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto inflate = [gap](cv::Rect r) {
        return cv::Rect(r.x - gap, r.y - gap, r.width + 2 * gap, r.height + 2 * gap);
    };

    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if ((inflate(dets[i].bbox) & inflate(dets[j].bbox)).area() > 0)
                parent[find(i)] = find(j);

    std::map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[find(i)].push_back(i);

    std::vector<Detection> merged;
    merged.reserve(groups.size());
    for (auto& [root, idx] : groups) {
        if (idx.size() == 1) { merged.push_back(dets[idx[0]]); continue; }
        Detection m = dets[idx[0]];
        cv::Rect box = dets[idx[0]].bbox;
        double wsum = dets[idx[0]].area;
        cv::Point2f csum = dets[idx[0]].centroid * dets[idx[0]].area;
        float asum = dets[idx[0]].area;
        for (size_t k = 1; k < idx.size(); ++k) {
            const Detection& d = dets[idx[k]];
            box |= d.bbox;                     // birleşik (union) kutu
            csum += d.centroid * d.area;       // alan-ağırlıklı centroid
            wsum += d.area;
            asum += d.area;
        }
        m.bbox = box;
        m.area = asum;
        m.centroid = wsum > 0 ? cv::Point2f(csum.x / static_cast<float>(wsum),
                                            csum.y / static_cast<float>(wsum))
                              : m.centroid;
        m.aspect_ratio = box.height > 0 ? static_cast<float>(box.width) / box.height : 1.f;
        merged.push_back(m);
    }
    dets.swap(merged);
}

// Her tespitin kutusunu, GÖK ÖNÜNDEKİ tam silüete genişletir.
// obj = ¬sky_raw; tespit merkezinin düştüğü bağlı bileşenin bbox'ı = cismin tamamı
// (hareketsiz iç gövde dahil). Bileşen aşırı büyükse (merkez zemine/ağaca düşmüş)
// hareket bbox'ı korunur.
void expand_to_silhouette(std::vector<Detection>& dets, const cv::Mat& sky_raw,
                          double max_frac) {
    cv::Mat obj;
    cv::bitwise_not(sky_raw, obj);                 // gök DEĞİL = cisim(ler)
    cv::Mat lbl, stats, cents;
    const int n = cv::connectedComponentsWithStats(obj, lbl, stats, cents, 8);
    if (n <= 1) return;
    const double max_area = max_frac * obj.total();

    for (Detection& d : dets) {
        const int cx = std::min(std::max(0, cvRound(d.centroid.x)), obj.cols - 1);
        const int cy = std::min(std::max(0, cvRound(d.centroid.y)), obj.rows - 1);
        const int L = lbl.at<int>(cy, cx);
        if (L <= 0) continue;                      // merkez gökte (koyu piksel yok)
        const double a = stats.at<int>(L, cv::CC_STAT_AREA);
        if (a > max_area) continue;                // dev bileşen (zemin) → güvenme
        d.bbox = cv::Rect(stats.at<int>(L, cv::CC_STAT_LEFT),
                          stats.at<int>(L, cv::CC_STAT_TOP),
                          stats.at<int>(L, cv::CC_STAT_WIDTH),
                          stats.at<int>(L, cv::CC_STAT_HEIGHT));
    }
}
}  // namespace

MovingTargetDetector::MovingTargetDetector() : MovingTargetDetector(Params{}) {}

MovingTargetDetector::MovingTargetDetector(Params params) : p_(params) {
    // detectShadows=false → maske 0/255 (gölge 127 kalabalığı yok, daha sade).
    mog_ = cv::createBackgroundSubtractorMOG2(p_.mog_history, p_.mog_var_thresh, false);
}

void MovingTargetDetector::reset() {
    prev_gray_.release();
    last_mask_.release();
    seen_ = 0;
    last_count_ = 0;
    mog_ = cv::createBackgroundSubtractorMOG2(p_.mog_history, p_.mog_var_thresh, false);
}

bool MovingTargetDetector::detect(const Frame& in, std::vector<Detection>& out) {
    out.clear();
    last_count_ = 0;
    const cv::Mat gray = to_gray(in.image);
    const cv::Size sz = gray.size();
    last_mask_ = cv::Mat::zeros(sz, CV_8U);

    // --- İlgi bölgesi (ROI): cue verilmişse o, yoksa tüm kare eksi kenar bandı.
    // Kenar bandı: warp'tan gelen kayan siyah şerit fark/MOG2'yi yanıltmasın.
    cv::Rect full(p_.border_margin, p_.border_margin,
                  sz.width - 2 * p_.border_margin, sz.height - 2 * p_.border_margin);
    cv::Rect roi = roi_.empty() ? full : (roi_ & full);
    if (roi.width <= 0 || roi.height <= 0) { prev_gray_ = gray; ++seen_; return false; }

    // --- MOG2 dalı: TÜM kare üzerinde uygula (model uzamsal tutarlı kalsın),
    // sonra ROI'yi al. Büyük/yavaş cisimleri (zeplin) dolu yakalar.
    cv::Mat fg_full;
    mog_->apply(gray, fg_full);                 // arka planı öğrenir + maske üretir
    cv::Mat mask_mog = fg_full(roi) > 127;      // 0/255

    ++seen_;

    // --- Kare-farkı dalı: ardışık kareler arası mutlak fark. Küçük/hızlı hedef.
    cv::Mat mask = mask_mog.clone();            // birleşik maske ROI boyutunda
    if (!prev_gray_.empty() && prev_gray_.size() == sz) {
        cv::Mat diff, mask_diff;
        cv::absdiff(gray(roi), prev_gray_(roi), diff);
        cv::threshold(diff, mask_diff, p_.diff_thresh, 255, cv::THRESH_BINARY);
        cv::bitwise_or(mask, mask_diff, mask);  // union: iki dalı birleştir
    }
    prev_gray_ = gray;

    // Model ısınana kadar tespitlere güvenme (MOG2 ilk karelerde her şeyi fg sayar).
    if (seen_ <= p_.warmup) return false;

    // --- Gökyüzü kapısı (sky gate): zemin paralaksını elemek için sky maskesi.
    //   Top-hat dalından ÖNCE hesaplanır → saliency yalnız gök içinde aranır.
    //   gökyüzü = parlak ∧ (soluk ∨ mavi). Yeşil/doygun zemin reddedilir.
    //   `sky`     = dilate'li → kapı (gate) ve top-hat maskesi için.
    //   `sky_raw` = dilate'siz → silüet (expand_to_silhouette) için.
    cv::Mat sky, sky_raw;
    if (p_.sky_gate && in.image.channels() == 3) {
        cv::Mat hsv, ch[3];
        cv::cvtColor(in.image, hsv, cv::COLOR_BGR2HSV);
        cv::split(hsv, ch);                              // ch[0]=H ch[1]=S ch[2]=V
        cv::Mat bright = ch[2] > p_.sky_v_min;
        cv::Mat pale   = ch[1] < p_.sky_s_max;
        cv::Mat blue   = (ch[0] > 95) & (ch[0] < 135);   // mavi ton aralığı
        cv::bitwise_or(pale, blue, sky);
        cv::bitwise_and(sky, bright, sky);
        sky_raw = sky.clone();                           // dilate'siz → silüet için
        if (p_.sky_dilate > 1) {
            const cv::Mat k = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, {p_.sky_dilate, p_.sky_dilate});
            cv::dilate(sky, sky, k);                      // dilate'li → kapı (gate) için
        }
    }

    // --- Doku haritası (#14): |gradyan| büyüklüğü → "pürüzsüz" (gök/haze) maskesi.
    //   Zemin (ağaç/çim) yüksek gradyan; gök/haze düşük. Aday halkası dokuluysa zemin.
    cv::Mat smooth;  // 0/255: pürüzsüz pikseller (yerel kenar-yoğunluğu düşük = gök/haze)
    if (p_.texture_gate) {
        cv::Mat gx, gy;
        cv::Sobel(gray, gx, CV_16S, 1, 0);
        cv::Sobel(gray, gy, CV_16S, 0, 1);
        cv::convertScaleAbs(gx, gx);
        cv::convertScaleAbs(gy, gy);
        cv::Mat mag;
        cv::addWeighted(gx, 0.5, gy, 0.5, 0.0, mag);   // ~|gradyan|
        cv::Mat edges = mag > p_.texture_thresh;       // 0/255 piksel-kenar
        // Yerel kenar YOĞUNLUĞU: ağaç/çim bölgesi yoğun kenar → yüksek; tek uçağın
        // kendi kenarı SEYREK → düşük (uçak korunur). Pürüzsüz = düşük yoğunluk.
        cv::Mat density;
        cv::boxFilter(edges, density, CV_8U, {p_.texture_win, p_.texture_win});
        smooth = density < p_.texture_density_max;
    }

    // --- Top-hat dalı: SOLUK/KÜÇÜK hedefi MEKÂNSAL kontrastla yakala.
    //   absdiff/MOG2 parlak zeplini parlak bulut önünde kaçırır (Δ küçük); ama
    //   cisim çevresinden ayrıktır. white-hat (çevreden parlak) ∪ black-hat
    //   (çevreden koyu) → iki yönü de yakala. Yalnız gök içinde (sky) uygula ki
    //   zemin/ağaç dokusu sel olmasın. Bu dal HER karede çalışır (hareket
    //   gerekmez) → soluk hedef kararlı blob verir, M-of-N onayını doldurabilir.
    if (p_.tophat && p_.tophat_ksize > 1) {
        const cv::Mat k = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, {p_.tophat_ksize, p_.tophat_ksize});
        cv::Mat sal;
        if (p_.tophat_mode == 1) {                            // yalnız KOYU (çevreden koyu cisim)
            cv::morphologyEx(gray, sal, cv::MORPH_BLACKHAT, k);
        } else if (p_.tophat_mode == 2) {                     // yalnız PARLAK
            cv::morphologyEx(gray, sal, cv::MORPH_TOPHAT, k);
        } else {                                              // her ikisi → en güçlüsü
            cv::Mat what, bhat;
            cv::morphologyEx(gray, what, cv::MORPH_TOPHAT,   k);  // gray - open  : çevreden parlak
            cv::morphologyEx(gray, bhat, cv::MORPH_BLACKHAT, k);  // close - gray : çevreden koyu
            cv::max(what, bhat, sal);                            // iki yönün en güçlüsü
        }
        cv::Mat sal_mask = sal > p_.tophat_thresh;            // 0/255, tam kare
        if (!sky.empty()) cv::bitwise_and(sal_mask, sky, sal_mask);  // yalnız gök içi
        cv::bitwise_or(mask, sal_mask(roi), mask);            // ROI'ye kırp, hareket maskesine ekle
    }

    // --- Morfoloji: aç (gürültü sil) → kapat (blob içini birleştir).
    if (p_.open_ksize > 1) {
        const cv::Mat k = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, {p_.open_ksize, p_.open_ksize});
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    }
    if (p_.close_ksize > 1) {
        const cv::Mat k = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, {p_.close_ksize, p_.close_ksize});
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    }

    // --- Connected components → aday blob'lar.
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    for (int i = 1; i < n; ++i) {                // 0 = arka plan, atla
        const double area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < p_.min_area || area > p_.max_area) continue;

        const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(i, cv::CC_STAT_TOP);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // ROI alt-matrisinde bulduk → global koordinata kaydır (roi.x, roi.y).
        const float gx = static_cast<float>(centroids.at<double>(i, 0) + roi.x);
        const float gy = static_cast<float>(centroids.at<double>(i, 1) + roi.y);

        // Gökyüzü kapısı (Issue #13): "ÇEVRE (ring) gök mü?" testi.
        // NEDEN merkez/öz-örtüşme değil: hava hedefi (koyu gondol, dron) KENDİ pikselleri
        // koyudur → gök sayılmaz; ama ÇEVRESİ gökle sarılıdır. Ufuk ağaç-tepesinin ise
        // üstü gök, ALTI zemindir → çevre gök oranı düşük. Blobu R piksel şişirip
        // (çevre halkası) bu halkanın ham-gök (sky_raw) oranına bakarız.
        //   gondol  → halka ~tamamı gök → geçer
        //   ağaç-tepesi → halkanın altı zemin → oran düşük → elenir
        if (!sky_raw.empty() && p_.sky_overlap_min > 0.0) {
            const int R = p_.sky_ring;
            cv::Rect gb(x + roi.x, y + roi.y, w, h);                       // global blob kutusu
            cv::Rect eb(gb.x - R, gb.y - R, gb.width + 2 * R, gb.height + 2 * R);
            eb &= cv::Rect(0, 0, sky_raw.cols, sky_raw.rows);             // kareye kırp
            const double ring = std::max(1.0, eb.area() - area);          // halka alanı (blob hariç)
            const double ring_sky = cv::countNonZero(sky_raw(eb));        // halkadaki gök (blob koyu→~0 katkı)
            if (ring_sky / ring < p_.sky_overlap_min) continue;
        }

        // Doku kapısı (#14): çevre ARKA PLANI dokulu ise (zemin) ele. Renk gate'inin
        // haze sızıntısını kapatır: haze pürüzsüz ama "gök renkli"; zemin dokulu.
        // ÖNEMLİ: halka blob kenarından texture_offset kadar ÖTEDE başlar → cismin
        // KENDİ kenarları yoğunluğa karışmaz (yoksa uçak da reddedilir). Ölçü =
        // dış kutu (bbox+ofset+halka) ile iç kutu (bbox+ofset) arasındaki HALKA.
        if (p_.texture_gate && !smooth.empty()) {
            const cv::Rect frame(0, 0, smooth.cols, smooth.rows);
            const int o = p_.texture_offset, R = p_.sky_ring;
            cv::Rect gb(x + roi.x, y + roi.y, w, h);
            cv::Rect inner = (cv::Rect(gb.x - o,     gb.y - o,     gb.width + 2 * o,       gb.height + 2 * o))       & frame;
            cv::Rect outer = (cv::Rect(gb.x - o - R, gb.y - o - R, gb.width + 2 * (o + R), gb.height + 2 * (o + R))) & frame;
            const double ring_area   = std::max(1, outer.area() - inner.area());
            const double ring_smooth = cv::countNonZero(smooth(outer)) - cv::countNonZero(smooth(inner));
            if (ring_smooth / ring_area < p_.texture_smooth_min) continue;
        }

        Detection d;
        d.centroid = cv::Point2f(gx, gy);
        d.bbox = cv::Rect(x + roi.x, y + roi.y, w, h);
        d.area = static_cast<float>(area);
        d.aspect_ratio = h > 0 ? static_cast<float>(w) / static_cast<float>(h) : 1.f;
        d.score = 0.f;                           // "drone olma" P3'te atanır
        d.t = in.t;
        out.push_back(d);
    }

    // Parça parça blob'ları tek hedefe topla (büyük cisim → tek aday).
    if (p_.merge_gap > 0) merge_detections(out, p_.merge_gap);

    // Kutuyu cismin tam silüetine genişlet (gök önünde ¬sky bileşeni).
    if (p_.tight_box && !sky_raw.empty()) expand_to_silhouette(out, sky_raw, p_.max_sil_frac);

    // Teşhis maskesini tam-kare yerleştir.
    mask.copyTo(last_mask_(roi));
    last_count_ = static_cast<int>(out.size());
    return true;
}

} // namespace dtrack
