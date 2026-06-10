#pragma once
// ============================================================================
//  LockIntegrity — kilitli hedefin "hâlâ doğru şey mi?" GEOMETRİK denetimi.
//
//  NEDEN AYRI BİR KATMAN?  Görsel tracker (NanoTrack/Siamese) "neye benziyor"u
//  tutar ama "doğru şeyi mi tutuyorum"u BİLEMEZ: düz gökte, yumuşak bulutta,
//  hatta yerdeki bir AĞACA kilitlenince de güveni 0.8+ kalır (hasat verisinde
//  gözle doğrulandı). Yani tracker güveni tek başına sürüklenmeyi yakalayamaz.
//
//  İŞE YARAYAN AYRAÇ = GEOMETRİ (görünüm değil). Bu projenin tekrarlanan dersi:
//    - Gerçek hava aracının kutu ÇEVRE HALKASI göktür; ağaç/tribünün çevresi yer.
//    - Gerçek hedef kutusu MAKUL boyutta kalır; tracker patlayınca tüm kareyi kaplar.
//    - Gerçek hedef kareden kareye YUMUŞAK gezer; yanlış eşleşme ışınlanır.
//
//  Üç bağımsız, saf, donanım-agnostik denetçi. Hepsi DAR ROI üzerinde birkaç bin
//  piksel → STM32N6'da Cortex-M55'te ~bedava (kalıcı tampon yok). Backbone NPU'da
//  int8 koşarken bu denetim M55'te paralel akar.
// ============================================================================
#include <opencv2/core.hpp>

namespace dtrack {

// Bütünlük denetiminin birleşik sonucu (GuidanceController doldurur; HUD/debug okur).
struct IntegrityResult {
    bool        ok     = true;   // üç eksen de geçti mi
    float       sky    = 1.f;    // ölçülen gök-çevre oranı [0,1]
    const char* reason = "";     // ilk düşen eksen: "" | "sky" | "size" | "motion"
};

namespace lock_integrity {

// Kutu ÇEVRE HALKASININ gök oranı [0,1]. Halka = kutuyu ring_px şişir, İÇ kutuyu çıkar.
//   Gök pikseli: BGR'de  B >= G  (çimen/ağaç yeşil-baskındır)  ∪  parlaklık > bright_thresh
//   (bulut/açık gök parlaktır). Gri görüntüde renk yok → yalnız parlaklık kullanılır.
// ring_px <= 0 verilirse max(w,h) kullanılır (kutuyla orantılı halka).
float sky_ring(const cv::Mat& frame, const cv::Rect& box,
               int ring_px = -1, float bright_thresh = 180.f);

// Boyut sağlığı: kutu alanı kare alanının > max_area_frac'i VEYA kilit kutusundan
// > max_growth× büyükse → patlama/sürüklenme → false. (lock_box0.area()<=0 ise büyüme atlanır.)
bool size_sane(const cv::Rect& box, const cv::Size& frame_size,
               const cv::Rect& lock_box0, float max_area_frac, float max_growth);

// Hareket sağlığı: merkez sıçraması > max_jump_px → ışınlanma/yanlış eşleşme → false.
// (prev_box.area()<=0 ise referans yok → true.)
bool motion_sane(const cv::Rect& box, const cv::Rect& prev_box, float max_jump_px);

// Kutu içi KENAR YOĞUNLUĞU [0,~1]: Sobel büyüklüğü ortalaması / 255.
//   Drone silüeti KESKİN kenarlı (yüksek), bulut tutamı YUMUŞAK (düşük) — gök-çevre
//   ve boyut bandını geçen bulutu reseed'de elemenin tek ayracı (hasat filtresinde
//   kanıtlandı: drone ~0.12+, bulut-kilidi ~0.05–0.10). Gri/renkli çalışır.
float edge_density(const cv::Mat& frame, const cv::Rect& box);

}  // namespace lock_integrity
}  // namespace dtrack
