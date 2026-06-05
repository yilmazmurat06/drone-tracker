#pragma once
//
// SceneModel: sentetik sahnenin YER-GERÇEĞİ (ground truth) modeli. Saf matematik;
// OpenCV'ye ihtiyaç duymaz, deterministiktir (aynı t -> aynı sonuç) ve bu yüzden
// birim testi kolaydır.
//
// Modelin amacı, kamera (görüntü) ile IMU (gyro) çıktısının AYNI fiziksel
// gerçekten türemesini sağlamak. Böylece stabilizasyon füzyonu (gyro + optical
// flow) gerçek bir problemi çözer, sahte bir şeyi değil.
//
// Fizik (basit pinhole + küçük açı yaklaşımı):
//   - Platform titreşimi -> kamera açısı theta(t) (yaw=x, pitch=y), sinüs toplamı.
//   - Gyro bunun TÜREVİNİ (açısal hız) ölçer: omega(t) = d theta/dt.
//   - Görüntüde her şey ego_shift = focal_px * theta(t) kadar kayar (arka plan VE hedef).
//   - Hedefin KENDİ hareketi world_traj(t) (düz hareket + manevra sinüsü).
//   - Gözlemlenen hedef konumu = world_traj(t) + ego_shift(t).
//   - Mükemmel stabilizasyon ego_shift'i kaldırır -> geriye world_traj(t) kalır.

#include <cmath>
#include <cstdint>

namespace dtrack::io {

struct Vec2 {
    float x{0};
    float y{0};
};

struct SceneConfig {
    // Görüntü ve optik
    int width = 640;
    int height = 512;
    float focal_px = 700.0f;  // odak uzaklığı (piksel); gyro açısı <-> piksel kayma

    // Ego-motion (platform titreşimi): iki eksende sinüs toplamı.
    // Genlik radyan; küçük (örn. 0.012 rad ~ 0.7°) ama focal*theta ~ 8 px kaymaya yeter.
    float ego_amp_yaw = 0.012f;    // x ekseni açı genliği (rad)
    float ego_amp_pitch = 0.009f;  // y ekseni açı genliği (rad)
    float ego_freq_yaw = 1.7f;     // Hz
    float ego_freq_pitch = 2.3f;   // Hz
    // İkinci (daha yüksek frekanslı, küçük) titreşim bileşeni -> daha gerçekçi.
    float ego_amp2 = 0.003f;
    float ego_freq2 = 11.0f;       // Hz (yüksek frekans titreşim)

    // Hedef yörüngesi (görüntü düzleminde, ego HARİÇ "gerçek" hareket).
    Vec2 target_start{120.0f, 300.0f};  // başlangıç piksel konumu
    Vec2 target_vel{55.0f, -18.0f};     // piksel/saniye (düz bileşen)
    float target_maneuver_amp = 40.0f;  // manevra genliği (piksel)
    float target_maneuver_freq = 0.25f; // Hz (yavaş zikzak)

    // Hedef görünümü
    float target_sigma_px = 1.1f;   // Gaussian yarıçap -> ~2-5 px görünür leke
    float target_intensity = 90.0f; // arka plana EKLENEN parlaklık (0-255 ölçeği)

    // Arka plan
    float bg_level = 35.0f;      // taban parlaklık
    float bg_texture = 18.0f;    // doku kontrastı (optical flow için köşe sağlar)

    uint32_t seed = 12345;
};

class SceneModel {
public:
    explicit SceneModel(const SceneConfig& cfg) : cfg_(cfg) {}

    const SceneConfig& config() const { return cfg_; }

    // Kamera açısı theta(t) [rad], (yaw=x, pitch=y).
    Vec2 ego_angle(double t) const {
        const double w1 = kTwoPi * cfg_.ego_freq_yaw * t;
        const double w2 = kTwoPi * cfg_.ego_freq_pitch * t;
        const double wv = kTwoPi * cfg_.ego_freq2 * t;
        return {static_cast<float>(cfg_.ego_amp_yaw * std::sin(w1) +
                                   cfg_.ego_amp2 * std::sin(wv)),
                static_cast<float>(cfg_.ego_amp_pitch * std::sin(w2) +
                                   cfg_.ego_amp2 * std::sin(wv * 0.83))};
    }

    // Gyro'nun ölçtüğü GERÇEK açısal hız omega(t) = d theta/dt [rad/s].
    // (z ekseni roll'u modellemiyoruz; 0.)
    Vec2 ego_rate(double t) const {
        const double w1 = kTwoPi * cfg_.ego_freq_yaw;
        const double w2 = kTwoPi * cfg_.ego_freq_pitch;
        const double wv = kTwoPi * cfg_.ego_freq2;
        return {static_cast<float>(cfg_.ego_amp_yaw * w1 * std::cos(w1 * t) +
                                   cfg_.ego_amp2 * wv * std::cos(wv * t)),
                static_cast<float>(cfg_.ego_amp_pitch * w2 * std::cos(w2 * t) +
                                   cfg_.ego_amp2 * wv * 0.83 * std::cos(wv * 0.83 * t))};
    }

    // Ego kaynaklı görüntü kayması [piksel] = focal * theta.
    Vec2 ego_shift_px(double t) const {
        const Vec2 a = ego_angle(t);
        return {cfg_.focal_px * a.x, cfg_.focal_px * a.y};
    }

    // Hedefin EGO HARİÇ gerçek görüntü konumu (yer-gerçeği; doğruluk ölçümü için).
    Vec2 target_truth_px(double t) const {
        const float tf = static_cast<float>(t);
        const float maneuver = cfg_.target_maneuver_amp *
            static_cast<float>(std::sin(kTwoPi * cfg_.target_maneuver_freq * t));
        return {cfg_.target_start.x + cfg_.target_vel.x * tf,
                cfg_.target_start.y + cfg_.target_vel.y * tf + maneuver};
    }

    // Kamerada GÖZLEMLENEN hedef konumu = gerçek hareket + ego kayması.
    Vec2 target_observed_px(double t) const {
        const Vec2 truth = target_truth_px(t);
        const Vec2 shift = ego_shift_px(t);
        return {truth.x + shift.x, truth.y + shift.y};
    }

private:
    static constexpr double kTwoPi = 6.283185307179586;
    SceneConfig cfg_;
};

}  // namespace dtrack::io
