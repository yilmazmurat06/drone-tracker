// ============================================================================
//  GyroFlowStabilizer implementasyonu — optik akış + gyro füzyonu.
// ============================================================================
#include "dtrack/stabilization/gyro_flow_stabilizer.hpp"

#include <cmath>

#include <opencv2/calib3d.hpp>   // cv::Rodrigues
#include <opencv2/imgproc.hpp>   // cv::warpPerspective

namespace dtrack {

GyroFlowStabilizer::GyroFlowStabilizer() : GyroFlowStabilizer(Params{}) {}
GyroFlowStabilizer::GyroFlowStabilizer(Params params)
    : p_(params), of_(params.of) {}

void GyroFlowStabilizer::reset() {
    of_.reset();
    has_prev_ = false;
    last_mode_ = Mode::None;
}

// K = [[f, 0, cx], [0, f, cy], [0, 0, 1]],  f = (w/2) / tan(FOV_h/2).
// Asal nokta görüntü merkezi varsayılır (sim için makul; gerçek kamera kalibre edilir).
cv::Matx33f GyroFlowStabilizer::intrinsics(const cv::Size& s) const {
    return intrinsics(s, p_.fov_h_deg);
}

cv::Matx33f GyroFlowStabilizer::intrinsics(const cv::Size& s, float fov_h_deg) const {
    const float w = static_cast<float>(s.width);
    const float h = static_cast<float>(s.height);
    const float half_fov = 0.5f * fov_h_deg * static_cast<float>(CV_PI) / 180.f;
    const float f = (w * 0.5f) / std::tan(half_fov);
    return cv::Matx33f(f, 0, w * 0.5f,
                       0, f, h * 0.5f,
                       0, 0, 1);
}

// H = K · R · K⁻¹,  R = Rodrigues(rotvec).
// rotvec [radyan] kamera çerçevesinde bir dönme; bu, görüntü düzleminde saf
// dönmenin yarattığı homografiyi verir (öteleme/paralaks YOK).
cv::Matx33f GyroFlowStabilizer::rotation_homography(const cv::Matx33f& K,
                                                    const cv::Vec3f& rotvec) {
    // DİKKAT: rvec'i CV_64F kur. Rodrigues çıkış derinliğini girişten alır;
    // float verirsek R de float olur ve aşağıda .at<double> ile okumak çöp verir.
    const cv::Mat rvec = (cv::Mat_<double>(3, 1) << rotvec[0], rotvec[1], rotvec[2]);
    cv::Mat R;
    cv::Rodrigues(rvec, R);                        // 3x3 CV_64F
    cv::Matx33f Rf;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            Rf(r, c) = static_cast<float>(R.at<double>(r, c));
    return K * Rf * K.inv();
}

// θ_cam = ∫ ω_cam dt  (ÖNCEKİ→GEÇERLİ kare arası dönme), trapez kuralıyla.
// ω_cam, AxisMap ile gövde gyro'sundan türetilir: ω_cam[k] = sign[k]·ω_body[src[k]].
cv::Vec3f GyroFlowStabilizer::integrate_rotation(const std::vector<Telemetry>& w) const {
    cv::Vec3f theta(0, 0, 0);
    if (w.size() < 2) return theta;

    // DİKKAT: Liftoff gyro_* DERECE/saniye birimindedir (rad/s DEĞİL — quaternion
    // çapraz-doğrulaması bunu ortaya çıkardı). Rodrigues radyan ister → çevir.
    constexpr float deg2rad = static_cast<float>(CV_PI) / 180.f;
    auto omega_cam = [&](const Telemetry& t) {
        const float body[3] = {static_cast<float>(t.gyro_pitch) * deg2rad,
                               static_cast<float>(t.gyro_roll) * deg2rad,
                               static_cast<float>(t.gyro_yaw) * deg2rad};
        cv::Vec3f o;
        for (int k = 0; k < 3; ++k)
            o[k] = p_.axes.sign[k] * body[p_.axes.src[k]];
        return o;
    };

    for (size_t i = 1; i < w.size(); ++i) {
        const float dt = static_cast<float>(w[i].t_rel - w[i - 1].t_rel);
        if (dt <= 0.f) continue;                 // sıralı değilse o aralığı atla
        theta += 0.5f * (omega_cam(w[i]) + omega_cam(w[i - 1])) * dt;  // trapez
    }
    return theta;
}

bool GyroFlowStabilizer::stabilize(const Frame& in,
                                   const std::vector<Telemetry>& window,
                                   Frame& out,
                                   cv::Matx33f& homography) {
    homography = cv::Matx33f::eye();

    // 1) Önce optik akışı dene (paralaksı da kapsar → güvenilirse en doğrusu).
    cv::Matx33f of_h;
    const bool of_ok = of_.stabilize(in, {}, out, of_h);
    if (of_ok) {
        homography = of_h;
        last_mode_ = Mode::Flow;
        has_prev_ = true;
        return true;
    }

    // İlk kare: hizalanacak önceki kare yok → kimlik. (out, of_ tarafından = in.)
    if (!has_prev_) {
        has_prev_ = true;
        last_mode_ = Mode::None;
        return false;
    }

    // 2) OF çöktü (özelliksiz gökyüzü) → gyro devralır.
    const cv::Vec3f theta = integrate_rotation(window);  // prev→cur dönme
    if (window.size() >= 2) {
        const cv::Matx33f K = intrinsics(in.image.size());
        // cur→prev hizalaması = prev→cur dönmenin TERSİ (−θ).
        homography = rotation_homography(K, -theta);
        // DİKKAT: of_ az önce out=in yaptı → out.image, in.image ile aynı veriyi
        // paylaşır. warpPerspective YERİNDE çalışamaz; ayrı bir hedefe yazıp ata.
        cv::Mat dst;
        cv::warpPerspective(in.image, dst, cv::Mat(homography),
                            in.image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);
        out.image = dst;
        out.id = in.id;
        out.t = in.t;
        last_mode_ = Mode::Gyro;
        return true;
    }

    // 3) Ne OF ne gyro → kimlik (out = in, of_ tarafından bırakıldı).
    last_mode_ = Mode::None;
    return false;
}

} // namespace dtrack
