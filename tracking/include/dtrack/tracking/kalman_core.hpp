#pragma once
//
// Kf2: 1 boyutlu SABİT HIZ (Constant Velocity, CV) Kalman filtresi çekirdeği.
// Durum: [konum, hız].  Saf C++ (OpenCV / proje başlığı YOK) -> birim testi
// OpenCV kurulmadan koşar ve GERÇEK tracker da bu çekirdeği kullanır.
//
// NEDEN 1B (2-durumlu) çekirdek, 4-durumlu (px,vx,py,vy) yerine:
//   Süreç matrisi F ve süreç gürültüsü Q eksen-bloklu (x ekseni px,vx; y ekseni
//   py,vy) ve ölçüm gürültüsü R KÖŞEGEN olduğunda, 4-durumlu CV filtresi İKİ
//   BAĞIMSIZ 2-durumlu filtreye ayrışır (x ve y birbirini etkilemez). Bu yüzden
//   tracker her iz için iki Kf2 (biri x, biri y) tutar. Sonuç: kapalı-form 2x2
//   matematik (genel matris kütüphanesi gerekmez), daha hızlı ve sınanması kolay.
//
// §2.7 DERSİ (PROJECT_REPORT): İlk Kalman denemesi küçük (2-6 px) hedefte
//   "P kovaryansı kararsız" + "init_vel_var=1e4 -> salınım" yüzünden terk edilip
//   α-β'ye geçilmişti. Bu çekirdek o sorunları ADRESLER:
//     1) JOSEPH-FORM kovaryans güncellemesi  -> P daima simetrik+pozitif kalır
//        (naif P=(I-KH)P float'ta negatife kayabilir; Joseph kaymaz).
//     2) ÖLÇÜLÜ başlangıç P0 (init'te hız varyansı küçük tutulur) -> salınım yok.
//     3) Süreç gürültüsü σ_a (manevra ivmesi) ve ölçüm gürültüsü σ_r FİZİKSEL
//        parametre; varsayılanlar kararlı-durum kazancını α-β (α≈0.55) ile
//        eşleştirir, yani en kötü ihtimalle α-β kadar iyi, üstüne:
//          - coast'ta P büyür -> kapı (gate) DOĞAL genişler (elle ayar gerekmez)
//          - değişken dt'yi (kare atlama) ilkesel olarak handle eder.

#include <cmath>

namespace dtrack::tracking {

// 1B sabit-hız Kalman: durum [p (konum, px), v (hız, px/s)], 2x2 kovaryans P.
struct Kf2 {
    double p{0.0}, v{0.0};                            // durum
    double P00{1.0}, P01{0.0}, P10{0.0}, P11{1.0};    // kovaryans (P01==P10 simetri)

    // Başlat. pos_var: konum varyansı (px²); vel_var: hız varyansı (px²/s²).
    // vel_var ÖLÇÜLÜ seçilmeli (bkz. §2.7) -> büyük değer ilk karelerde salınım yapar.
    void init(double pos, double vel, double pos_var, double vel_var) {
        p = pos; v = vel;
        P00 = pos_var; P01 = 0.0; P10 = 0.0; P11 = vel_var;
    }

    // PREDICT: x = F x ;  P = F P Fᵀ + Q.
    //   F = [[1, dt],[0, 1]]  (sabit hız)
    //   Q = q · [[dt⁴/4, dt³/2],[dt³/2, dt²]]  (beyaz-gürültü ivme modeli, Bar-Shalom)
    //   q = σ_a²  (ivme spektral yoğunluğu, (px/s²)²).
    void predict(double dt, double q) {
        p += v * dt;
        // F P Fᵀ  (kapalı form):
        const double a = P00 + dt * (P01 + P10) + dt * dt * P11;
        const double b = P01 + dt * P11;
        const double c = P10 + dt * P11;
        const double d = P11;
        // + Q:
        const double dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt2 * dt2;
        P00 = a + q * dt4 * 0.25;
        P01 = b + q * dt3 * 0.5;
        P10 = c + q * dt3 * 0.5;
        P11 = d + q * dt2;
    }

    // Innovation (artık): y = z - p.  S = P00 + r (innovation kovaryansı).
    // NIS (Normalized Innovation Squared) = y²/S -> 1 serbestlik dereceli χ² (eksen başına).
    double innovation(double z) const { return z - p; }
    double innov_cov(double r) const { return P00 + r; }
    double nis(double z, double r) const {
        const double y = z - p;
        return (y * y) / (P00 + r);
    }

    // CORRECT (Joseph-form, sayısal kararlı):
    //   K = P Hᵀ S⁻¹,  H = [1, 0]
    //   x = x + K y
    //   P = (I - K H) P (I - K H)ᵀ + K R Kᵀ
    // r: ölçüm varyansı σ_r² (px²).
    void correct(double z, double r) {
        const double s = P00 + r;
        const double k0 = P00 / s;   // konum kazancı (kararlı-durumda ≈ α)
        const double k1 = P10 / s;   // hız kazancı
        const double y = z - p;
        p += k0 * y;
        v += k1 * y;

        // Joseph form. A = (I-KH) = [[1-k0, 0],[-k1, 1]].
        const double ic = 1.0 - k0;
        // M = A P
        const double m00 = ic * P00;
        const double m01 = ic * P01;
        const double m10 = P10 - k1 * P00;
        const double m11 = P11 - k1 * P01;
        // P' = M Aᵀ + K R Kᵀ   (R=r skaler, K R Kᵀ = r·[[k0²,k0k1],[k0k1,k1²]])
        const double nP00 = m00 * ic + r * k0 * k0;
        const double nP01 = -k1 * m00 + m01 + r * k0 * k1;
        const double nP10 = m10 * ic + r * k0 * k1;
        const double nP11 = -k1 * m10 + m11 + r * k1 * k1;
        P00 = nP00; P01 = nP01; P10 = nP10; P11 = nP11;
    }

    // PDAF (Probabilistic Data Association Filter) güncellemesi — TEK eksen.
    //
    // NEDEN: clutter'da (kapı içinde birden çok aday) "en yakını seç" (nearest
    // neighbor) yanlış adaya kilitlenip izi saptırabilir. PDAF bunun yerine kapı
    // içindeki TÜM adayları olabilirliklerine göre AĞIRLIKLANDIRIP birleşik bir
    // innovation üretir (Bar-Shalom). Tek hedef + clutter için NN'den belirgin
    // üstün; az clutter'da NN'e yakınsar (β tek adayda ~1 olur).
    //
    // Bu eksen için: birleşik innovation  ỹ = Σ_i β_i y_i ;  β0 = "geçerli ölçüm yok".
    //   x      += K·ỹ
    //   P^c     = P_pred − K S Kᵀ                 (tek-ölçüm güncellenmiş kovaryans)
    //   P̃      = K·(Σ_i β_i y_i² − ỹ²)·Kᵀ        (innovation yayılımı; β0 ile büyür)
    //   P(k|k)  = β0·P_pred + (1−β0)·P^c + P̃     (moment eşleme)
    // β_i ağırlıkları İKİ eksenin ortak 2B olabilirliğinden gelir (tracker hesaplar);
    // köşegen R + ayrışık eksen sayesinde GÜNCELLEME eksen-bazında yapılabilir.
    void correct_pda(const double* y, const double* beta, int m, double beta0, double r) {
        const double s = P00 + r;
        const double k0 = P00 / s;
        const double k1 = P10 / s;
        double ytil = 0.0, sum_by2 = 0.0;
        for (int i = 0; i < m; ++i) {
            ytil += beta[i] * y[i];
            sum_by2 += beta[i] * y[i] * y[i];
        }
        p += k0 * ytil;
        v += k1 * ytil;
        const double Pp00 = P00, Pp01 = P01, Pp10 = P10, Pp11 = P11;  // P_pred
        // P^c = P_pred − K S Kᵀ   (K S Kᵀ = s·[[k0²,k0k1],[k0k1,k1²]])
        const double Pc00 = P00 - s * k0 * k0;
        const double Pc01 = P01 - s * k0 * k1;
        const double Pc10 = P10 - s * k1 * k0;
        const double Pc11 = P11 - s * k1 * k1;
        const double spread = sum_by2 - ytil * ytil;  // ≥ 0
        P00 = beta0 * Pp00 + (1.0 - beta0) * Pc00 + k0 * k0 * spread;
        P01 = beta0 * Pp01 + (1.0 - beta0) * Pc01 + k0 * k1 * spread;
        P10 = beta0 * Pp10 + (1.0 - beta0) * Pc10 + k1 * k0 * spread;
        P11 = beta0 * Pp11 + (1.0 - beta0) * Pc11 + k1 * k1 * spread;
    }
};

}  // namespace dtrack::tracking
