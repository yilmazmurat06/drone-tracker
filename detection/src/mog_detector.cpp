#include "dtrack/detection/mog_detector.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace dtrack::detection {

MogDetector::MogDetector(DetectorConfig cfg) : cfg_(cfg) {
    mog_ = cv::createBackgroundSubtractorMOG2(cfg_.mog_history, cfg_.mog_var_threshold,
                                              /*detectShadows=*/false);
    const int k = cfg_.tophat_ksize;
    tophat_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {k, k});
    tophat_kernel_small_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    tophat_kernel_med_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {9, 9});
    tophat_kernel_large_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15});
    const int d = std::max(1, cfg_.fg_dilate);
    fg_kernel_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, {2 * d + 1, 2 * d + 1});
}

void MogDetector::reset() {
    mog_ = cv::createBackgroundSubtractorMOG2(cfg_.mog_history, cfg_.mog_var_threshold,
                                              false);
    h_cum_ = cv::Matx33f::eye();
    has_ref_ = false;
    resets_ = 0;
    cue_ = common::TargetCue{};
}

void MogDetector::make_reference(const cv::Mat& gray) {
    // Güncel kareyi yeni referans yap: kümülatif dönüşüm kimlik, model sıfırlanır.
    h_cum_ = cv::Matx33f::eye();
    ref_size_ = gray.size();
    mog_->clear();
    has_ref_ = true;
}

std::vector<common::Detection> MogDetector::detect(const common::StabilizedFrame& sf) {
    std::vector<common::Detection> out;
    if (!sf.frame || sf.frame->image.empty()) return out;
    const cv::Mat& gray = sf.frame->image;

    // --- 1) Kümülatif referans kaydını güncelle. ---
    if (!has_ref_) {
        make_reference(gray);
    } else {
        // curr->ref = (prev->ref) ∘ (curr->prev) = h_cum_ * H_inter.
        h_cum_ = h_cum_ * sf.ego.homography;
        const float tx = h_cum_(0, 2), ty = h_cum_(1, 2);
        const double reset_lim =
            cfg_.ref_reset_frac * std::min(ref_size_.width, ref_size_.height);
        if (std::hypot(tx, ty) > reset_lim) {
            make_reference(gray);
            ++resets_;
        }
    }

    // --- 2) Güncel kareyi referans koordinatına warp et (+ geçerli bölge maskesi). ---
    cv::Matx23f M(h_cum_(0, 0), h_cum_(0, 1), h_cum_(0, 2), h_cum_(1, 0), h_cum_(1, 1),
                  h_cum_(1, 2));
    cv::Mat reg, valid;
    cv::warpAffine(gray, reg, M, ref_size_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
    cv::warpAffine(cv::Mat(gray.size(), CV_8UC1, cv::Scalar(255)), valid, M, ref_size_,
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, 0);

    // --- 3) Top-hat (uzamsal saliency) — 4 ölçek: 3×3 + 5×5 + 9×9 + 15×15. ---
    cv::Mat tophat;
    cv::Mat tophat5;  // eşik hesabı için 5×5 referans
    if (cfg_.multi_scale_tophat) {
        cv::Mat th3, th5, th9, th15;
        cv::morphologyEx(reg, th3, cv::MORPH_TOPHAT, tophat_kernel_small_);
        cv::morphologyEx(reg, th5, cv::MORPH_TOPHAT, tophat_kernel_);
        cv::morphologyEx(reg, th9, cv::MORPH_TOPHAT, tophat_kernel_med_);
        cv::morphologyEx(reg, th15, cv::MORPH_TOPHAT, tophat_kernel_large_);
        cv::Mat tmp = cv::max(th3, th5);
        tmp = cv::max(tmp, th9);
        tophat = cv::max(tmp, th15);
        th5.copyTo(tophat5);
    } else {
        cv::morphologyEx(reg, tophat, cv::MORPH_TOPHAT, tophat_kernel_);
        tophat.copyTo(tophat5);
    }

    // --- 3b) DoG — iki bant: küçük (uzak) + orta (yakın) hedef. ---
    if (cfg_.use_dog) {
        cv::Mat g1, g2, dog_small, dog_med, dog;
        // Küçük bant (2-6 px)
        cv::GaussianBlur(reg, g1, cv::Size(0, 0), cfg_.dog_sigma1);
        cv::GaussianBlur(reg, g2, cv::Size(0, 0), cfg_.dog_sigma2);
        dog_small = g1 - g2;
        // Orta bant (5-15 px)
        cv::GaussianBlur(reg, g1, cv::Size(0, 0), cfg_.dog_sigma1b);
        cv::GaussianBlur(reg, g2, cv::Size(0, 0), cfg_.dog_sigma2b);
        dog_med = g1 - g2;
        // İki bandın maksimumu.
        cv::max(dog_small, dog_med, dog);

        double dog_max;
        cv::minMaxLoc(dog, nullptr, &dog_max);
        if (dog_max > 0) {
            cv::Mat dog_norm;
            dog.convertTo(dog_norm, CV_8U, 255.0 / dog_max);
            double th_max;
            cv::minMaxLoc(tophat5, nullptr, &th_max);
            if (th_max > 0) {
                const float scale = static_cast<float>(th_max / 255.0) * 0.5f;
                cv::addWeighted(tophat, 1.0f, dog_norm, scale, 0, tophat, CV_8U);
            }
        }
    }

    // --- 4) MOG2 (zamansal hareket). ---
    cv::Mat fg;
    mog_->apply(reg, fg, cfg_.mog_learning_rate);
    cv::dilate(fg, fg, fg_kernel_);

    // --- 5) Eşik + birleştir: top-hat-parlak ∧ MOG2-hareketli ∧ geçerli. ---
    // Eşik 5×5 top-hat'ın istatistiklerinden hesaplanır (daha kararlı, gürültü az).
    // Çoklu ölçek maskesi (max) üzerinden uygulanır.
    cv::Scalar mean, stddev;
    cv::meanStdDev(tophat5, mean, stddev, valid);
    const double thr =
        std::max(mean[0] + cfg_.thresh_k * stddev[0], cfg_.thresh_min_abs);
    cv::Mat th_mask;
    cv::threshold(tophat, th_mask, thr, 255, cv::THRESH_BINARY);

    cv::Mat combined;
    cv::bitwise_and(th_mask, fg, combined);
    cv::bitwise_and(combined, valid, combined);
    // Tek-piksel gürültü çöpünü ele (3x3 açma); gerçek hedef bloku hayatta kalır.
    static const cv::Mat open_k =
        cv::getStructuringElement(cv::MORPH_ELLIPSE, {3, 3});
    cv::morphologyEx(combined, combined, cv::MORPH_OPEN, open_k);

    // --- 6) Bağlı bileşenler -> blob -> geometrik eleme -> alt-piksel centroid. ---
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(combined, labels, stats, centroids, 8);

    const cv::Matx33f h_inv = h_cum_.inv();  // referans -> güncel (geri eşleme)

    for (int lab = 1; lab < n; ++lab) {
        const double area = stats.at<int>(lab, cv::CC_STAT_AREA);
        if (area < cfg_.min_area || area > cfg_.max_area) continue;

        const int bx = stats.at<int>(lab, cv::CC_STAT_LEFT);
        const int by = stats.at<int>(lab, cv::CC_STAT_TOP);
        const int bw = stats.at<int>(lab, cv::CC_STAT_WIDTH);
        const int bh = stats.at<int>(lab, cv::CC_STAT_HEIGHT);
        const float aspect =
            static_cast<float>(std::max(bw, bh)) / std::max(1, std::min(bw, bh));
        if (aspect > cfg_.max_aspect) continue;

        // Top-hat yanıtıyla ağırlıklı alt-piksel centroid (referans koord).
        double sw = 0, sx = 0, sy = 0;
        float peak = 0;        // ham görüntü tepe parlaklığı
        float peak_th = 0;     // top-hat tepe yanıtı (zayıf çöp elemesi için)
        double inner_sum = 0;  // blob içi ham parlaklık toplamı (LCM iç bölgesi)
        int inner_cnt = 0;
        for (int y = by; y < by + bh; ++y) {
            const int* lrow = labels.ptr<int>(y);
            const uchar* trow = tophat.ptr<uchar>(y);
            const uchar* rrow = reg.ptr<uchar>(y);
            for (int x = bx; x < bx + bw; ++x) {
                if (lrow[x] != lab) continue;
                const double w = trow[x];
                sw += w;
                sx += w * x;
                sy += w * y;
                peak = std::max(peak, static_cast<float>(rrow[x]));
                peak_th = std::max(peak_th, static_cast<float>(trow[x]));
                inner_sum += rrow[x];
                ++inner_cnt;
            }
        }
        if (sw <= 0) continue;
        if (peak_th < cfg_.min_peak_tophat) continue;  // zayıf gürültü çöpü
        const float rx = static_cast<float>(sx / sw);
        const float ry = static_cast<float>(sy / sw);

        // --- Yönlü yerel kontrast (LCM/WLDM) son-elemesi ---
        if (cfg_.use_lcm && inner_cnt > 0) {
            const float inner_mean = static_cast<float>(inner_sum / inner_cnt);
            // Arka plan halkası yarıçapı: blob yarıçapı + boşluk.
            const int R = std::max(bw, bh) / 2 + cfg_.lcm_gap_px;
            const int p = cfg_.lcm_bg_patch;
            // 8 yön: ilk 4 ile son 4 KARŞIT çiftler (dirs[k] ile dirs[k+4]).
            static const int dirs[8][2] = {{1, 0},  {1, 1},  {0, 1},  {-1, 1},
                                           {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
            float bg[8];
            for (int k = 0; k < 8; ++k) {
                const int cx = static_cast<int>(std::lround(rx)) + dirs[k][0] * R;
                const int cy = static_cast<int>(std::lround(ry)) + dirs[k][1] * R;
                double s = 0;
                int c = 0;
                for (int yy = cy - p; yy <= cy + p; ++yy) {
                    if (yy < 0 || yy >= reg.rows) continue;
                    const uchar* rr = reg.ptr<uchar>(yy);
                    for (int xx = cx - p; xx <= cx + p; ++xx) {
                        if (xx < 0 || xx >= reg.cols) continue;
                        s += rr[xx];
                        ++c;
                    }
                }
                bg[k] = c > 0 ? static_cast<float>(s / c) : 0.0f;
            }
            // Belirginlik (prominence) = iç_ort - komşuların ortalaması.
            float bg_sum = 0;
            for (float v : bg) bg_sum += v;
            const float prominence = inner_mean - bg_sum / 8.0f;
            // Yeterince belirgin değilse (dim/düşük-SNR) LCM uygulama -> recall korunur.
            if (prominence >= cfg_.lcm_min_prominence) {
                // Karşıt çiftlerden biri bile her iki ucta iç parlaklığın
                // lcm_line_ratio katından parlaksa -> içinden çizgi/kenar geçiyor -> ele.
                const float line_thr = inner_mean * cfg_.lcm_line_ratio;
                bool is_line = false;
                for (int k = 0; k < 4; ++k) {
                    if (bg[k] >= line_thr && bg[k + 4] >= line_thr) { is_line = true; break; }
                }
                if (is_line) continue;
            }
        }

        // Referans -> güncel kare koordinatına geri eşle.
        const cv::Vec3f p = h_inv * cv::Vec3f(rx, ry, 1.0f);
        const float inv_w = (std::abs(p[2]) > 1e-6f) ? 1.0f / p[2] : 1.0f;

        common::Detection d;
        d.stamp = sf.frame->stamp;
        d.modality = sf.frame->modality;
        d.centroid = {p[0] * inv_w, p[1] * inv_w};
        d.area_px = static_cast<float>(area);
        d.aspect_ratio = aspect;
        d.intensity = peak;
        d.drone_score = 0.0f;  // skorlama Problem 3'te (IDiscriminator)
        out.push_back(d);
    }

    // --- 7) Kapalı-döngü cued ROI kurtarma (track-before-detect-lite). ---
    // Tracker kilitliyken (cue geçerli) ve global pass tahmin kapısı içinde aday
    // bulamadıysa, tahmin etrafındaki ROI'de DÜŞÜK eşikle (MOG2-AND'siz) tek en iyi
    // top-hat tepesini kurtar. Kapı içinde zaten tespit varsa kurtarma gereksiz.
    if (cfg_.use_cue_recovery && cue_.valid && has_ref_) {
        const float gate =
            std::clamp(cue_.gate_radius, static_cast<float>(cfg_.cue_min_radius),
                       static_cast<float>(cfg_.cue_max_radius));
        bool covered = false;
        for (const auto& d : out) {
            const float dx = d.centroid.x - cue_.predicted.x;
            const float dy = d.centroid.y - cue_.predicted.y;
            if (dx * dx + dy * dy <= gate * gate) { covered = true; break; }
        }
        if (!covered) {
            // Cue GÜNCEL koordinatta; tespit referans koordinatında. h_cum_ ile taşı.
            const cv::Vec3f cr =
                h_cum_ * cv::Vec3f(cue_.predicted.x, cue_.predicted.y, 1.0f);
            const float cwn = (std::abs(cr[2]) > 1e-6f) ? 1.0f / cr[2] : 1.0f;
            const float cxr = cr[0] * cwn, cyr = cr[1] * cwn;

            const int x0 = std::max(0, static_cast<int>(std::floor(cxr - gate)));
            const int y0 = std::max(0, static_cast<int>(std::floor(cyr - gate)));
            const int x1 = std::min(reg.cols - 1, static_cast<int>(std::ceil(cxr + gate)));
            const int y1 = std::min(reg.rows - 1, static_cast<int>(std::ceil(cyr + gate)));
            if (x1 > x0 && y1 > y0) {
                const cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
                // ROI lokal eşik: 5×5 top-hat istatistiğinden (yalnız geçerli pikseller).
                cv::Scalar m, s;
                cv::meanStdDev(tophat5(roi), m, s, valid(roi));
                const double local_thr = std::max(m[0] + cfg_.cue_thresh_k * s[0],
                                                  static_cast<double>(cfg_.cue_min_peak_tophat));
                // ROI'de en yüksek (çoklu-ölçek) top-hat tepesi — dairesel kapı + geçerli.
                float best_resp = -1.0f;
                int bxp = -1, byp = -1;
                for (int y = y0; y <= y1; ++y) {
                    const uchar* trow = tophat.ptr<uchar>(y);
                    const uchar* vrow = valid.ptr<uchar>(y);
                    for (int x = x0; x <= x1; ++x) {
                        if (!vrow[x]) continue;
                        const float dgx = x - cxr, dgy = y - cyr;
                        if (dgx * dgx + dgy * dgy > gate * gate) continue;
                        if (trow[x] > best_resp) { best_resp = trow[x]; bxp = x; byp = y; }
                    }
                }
                if (bxp >= 0 && best_resp >= local_thr) {
                    // Tepe etrafında 5×5 pencerede top-hat-ağırlıklı alt-piksel centroid.
                    const int w = 2;
                    double sw = 0, sx = 0, sy = 0, ar = 0;
                    float peak = 0;
                    for (int y = std::max(0, byp - w); y <= std::min(reg.rows - 1, byp + w); ++y) {
                        const uchar* trow = tophat.ptr<uchar>(y);
                        const uchar* rrow = reg.ptr<uchar>(y);
                        for (int x = std::max(0, bxp - w); x <= std::min(reg.cols - 1, bxp + w); ++x) {
                            const double wv = trow[x];
                            if (wv < local_thr) continue;
                            sw += wv;
                            sx += wv * x;
                            sy += wv * y;
                            ar += 1.0;
                            peak = std::max(peak, static_cast<float>(rrow[x]));
                        }
                    }
                    if (sw > 0) {
                        const float rx = static_cast<float>(sx / sw);
                        const float ry = static_cast<float>(sy / sw);
                        const cv::Vec3f pc = h_inv * cv::Vec3f(rx, ry, 1.0f);
                        const float iw = (std::abs(pc[2]) > 1e-6f) ? 1.0f / pc[2] : 1.0f;
                        common::Detection d;
                        d.stamp = sf.frame->stamp;
                        d.modality = sf.frame->modality;
                        d.centroid = {pc[0] * iw, pc[1] * iw};
                        d.area_px = std::max(1.0f, static_cast<float>(ar));
                        d.aspect_ratio = 1.0f;
                        d.intensity = peak;
                        d.drone_score = 0.0f;
                        d.meas_std = cfg_.cue_meas_std;  // daha az güven -> tracker daha büyük R kullanır
                        d.from_cue = true;
                        out.push_back(d);
                    }
                }
            }
        }
    }

    return out;
}

}  // namespace dtrack::detection
