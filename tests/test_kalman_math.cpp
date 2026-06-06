// Kalman çekirdeği (Kf2) SAF MATEMATİK doğrulama testi — OpenCV GEREKMEZ.
//
// Bu test, gerçek tracker'ın kullandığı `kalman_core.hpp` (Kf2) başlığını birebir
// içerir ve §2.7'de Kalman'ın terk edilmesine yol açan sorunların GİDERİLDİĞİNİ
// kanıtlar. Derleme (OpenCV'siz):
//   g++ -std=c++17 -O2 -Itracking/include tests/test_kalman_math.cpp -o tkm && ./tkm
//
// Sınananlar:
//   A) KARARLILIK (§2.7'nin "P kovaryansı kararsız" endişesi): her adımda P
//      simetrik + pozitif-tanımlı (P00>0, P11>0, det>0) kalır; NaN/Inf yok.
//   B) KARARLI-DURUM KAZANCI: konum kazancı k0 ≈ α (≈0.55) bandına oturur ->
//      Kalman, kanıtlanmış α-β ile aynı pürüzsüzlükte (en kötü = α-β).
//   C) DOĞRULUK: gürültülü sabit-hız + sinüs manevra yörüngesinde konum RMS düşük.
//   D) COAST: ölçüm kesilince P büyür (kapı doğal genişler), sonlu kalır, salınmaz.
//   E) FPS-DEĞİŞMEZLİK: 30/60/120 fps'te raporlanan hız ~aynı (fiziksel hız fps'ten bağımsız).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "dtrack/tracking/kalman_core.hpp"

using dtrack::tracking::Kf2;

static int g_failures = 0;
#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) {                                               \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__); \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// Fiziksel parametreler (tracker varsayılanlarıyla aynı).
static constexpr double kMeasStd    = 1.5;     // σ_r (px)
static constexpr double kAccelStd   = 1500.0;  // σ_a (px/s²)
static const double     kR          = kMeasStd * kMeasStd;
static const double     kQ          = kAccelStd * kAccelStd;

static bool spd_ok(const Kf2& k) {
    const double det = k.P00 * k.P11 - k.P01 * k.P10;
    return std::isfinite(k.P00) && std::isfinite(k.P11) && std::isfinite(k.P01) &&
           k.P00 > 0.0 && k.P11 > 0.0 && det > 0.0;
}

int main() {
    std::printf("test_kalman_math (Kf2 cekirdek, OpenCV'siz)\n");

    // === A+B+C: 60fps gürültülü CV+manevra; kararlılık, kazanç, RMS. ===
    {
        const double fps = 60.0, dt = 1.0 / fps;
        std::mt19937 rng(12345);
        std::normal_distribution<double> noise(0.0, kMeasStd);

        Kf2 k;
        // Tek-nokta başlatma: konum ölçüldü (var=R), hız bilinmiyor ama ÖLÇÜLÜ
        // başlangıç varyansı (σ_v0=60 px/s) -> §2.7 salınımı önlenir.
        const double p0_true = 100.0;
        k.init(p0_true + noise(rng), 0.0, kR, 60.0 * 60.0);

        double t = 0.0, sum_e2 = 0.0;
        int n = 0, spd_fail = 0;
        double last_k0 = 0.0;
        const int N = 600;
        for (int i = 0; i < N; ++i) {
            t += dt;
            // Gerçek yörünge: 80 px/s taban hız + sinüs manevra (genlik 40 px).
            const double truth = p0_true + 80.0 * t + 40.0 * std::sin(2.0 * t);

            k.predict(dt, kQ);
            if (!spd_ok(k)) ++spd_fail;
            // Predict sonrası (correct öncesi) kazanç:
            last_k0 = k.P00 / (k.P00 + kR);

            const double z = truth + noise(rng);
            k.correct(z, kR);
            if (!spd_ok(k)) ++spd_fail;

            if (i > 150) {  // warmup sonrası RMS
                const double e = k.p - truth;
                sum_e2 += e * e;
                ++n;
            }
        }
        const double rms = std::sqrt(sum_e2 / n);
        std::printf("   [A] kararlilik: P-SPD ihlali=%d (ideal 0)\n", spd_fail);
        std::printf("   [B] kararli-durum konum kazanci k0=%.3f (α-β'de α≈0.55)\n", last_k0);
        std::printf("   [C] konum RMS=%.3f px (σ_r=%.1f)\n", rms, kMeasStd);
        CHECK(spd_fail == 0);                       // A: hiç kararsızlık yok
        CHECK(last_k0 > 0.40 && last_k0 < 0.70);    // B: α≈0.55 bandı
        CHECK(rms < 2.0 * kMeasStd);                // C: ölçüm gürültüsü mertebesinde
    }

    // === D: COAST — ölçüm kesilince P büyür, sonlu kalır, NaN yok. ===
    {
        const double dt = 1.0 / 60.0;
        Kf2 k;
        k.init(0.0, 50.0, kR, 30.0 * 30.0);
        // Önce birkaç ölçümle otur.
        double truth = 0.0;
        for (int i = 0; i < 30; ++i) {
            truth += 50.0 * dt;
            k.predict(dt, kQ);
            k.correct(truth, kR);
        }
        const double P00_locked = k.P00;
        // Şimdi 30 kare COAST (ölçüm yok) -> sadece predict.
        bool monotonic_grow = true, finite = true;
        double prev = k.P00;
        for (int i = 0; i < 30; ++i) {
            k.predict(dt, kQ);
            if (!(k.P00 >= prev - 1e-9)) monotonic_grow = false;  // belirsizlik artmalı
            if (!std::isfinite(k.P00)) finite = false;
            prev = k.P00;
        }
        std::printf("   [D] coast: P00 kilitli=%.2f -> 30 kare coast=%.2f  (artan=%d, sonlu=%d)\n",
                    P00_locked, k.P00, (int)monotonic_grow, (int)finite);
        CHECK(monotonic_grow);          // coast'ta kapı doğal genişler
        CHECK(finite);                  // sonsuza patlamaz
        CHECK(k.P00 > P00_locked);      // coast belirsizliği kilitten büyük
    }

    // === E: FPS-DEĞİŞMEZLİK — 30/60/120'de raporlanan hız ~aynı. ===
    {
        auto run = [](double fps) -> double {
            const double dt = 1.0 / fps;
            std::mt19937 rng(777);
            std::normal_distribution<double> noise(0.0, kMeasStd);
            Kf2 k;
            k.init(0.0, 0.0, kR, 60.0 * 60.0);
            double t = 0.0, sum_v = 0.0;
            int n = 0;
            const int frames = static_cast<int>(3.0 * fps);  // 3 saniye
            for (int i = 0; i < frames; ++i) {
                t += dt;
                const double truth = 90.0 * t;  // sabit 90 px/s
                k.predict(dt, kQ);
                k.correct(truth + noise(rng), kR);
                if (t > 1.0) { sum_v += k.v; ++n; }  // 1s sonrası otur
            }
            return n ? sum_v / n : 0.0;
        };
        const double v30 = run(30.0), v60 = run(60.0), v120 = run(120.0);
        const double mx = std::max({v30, v60, v120});
        const double mn = std::min({v30, v60, v120});
        const double spread = mn > 1.0 ? mx / mn : 1e9;
        std::printf("   [E] hiz raporu: 30fps=%.1f 60fps=%.1f 120fps=%.1f px/s  yayilim=%.3f (ideal~1, gercek=90)\n",
                    v30, v60, v120, spread);
        CHECK(v30 > 70 && v30 < 110);
        CHECK(v60 > 70 && v60 < 110);
        CHECK(v120 > 70 && v120 < 110);
        CHECK(spread < 1.20);  // fps'ten ~bağımsız
    }

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
