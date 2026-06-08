#pragma once
// ============================================================================
//  SpscRingBuffer — Lock-free SPSC (Single-Producer / Single-Consumer) kuyruğu
// ============================================================================
//
//  NE İŞE YARAR?
//  Pipeline'da aşamalar ayrı thread'lerde koşunca (Adım 7), bir aşama veri
//  ÜRETİR, sonraki aşama TÜKETİR. Aralarında bir kuyruk lazım. Klasik kuyruk
//  mutex (kilit) kullanır → thread'ler kilidi beklerken durur. Gerçek-zamanlı
//  sistemde bu "duraklama" kabul edilemez.
//
//  "LOCK-FREE" = kilit yok. Üretici ve tüketici, paylaşılan iki sayaç
//  (head, tail) üzerinden std::atomic ile haberleşir; kimse beklemez.
//
//  "SPSC" = TAM bir üretici thread + TAM bir tüketici thread. Bu kısıt sayesinde
//  tasarım çok basit ve hızlı olur. (Çok üretici/çok tüketici çok daha zordur;
//  bizim pipeline'da aşamalar zincir olduğu için SPSC yeterli.)
//
//  NASIL ÇALIŞIR? (döngüsel/circular dizi)
//    - buffer_  : sabit boyutlu dizi (Capacity elemanı)
//    - head_    : üreticinin yazacağı bir sonraki indeks
//    - tail_    : tüketicinin okuyacağı bir sonraki indeks
//    - head_ == tail_            → BOŞ
//    - (head_+1) == tail_ (mod N)→ DOLU   (1 slot kasten boş bırakılır ki
//                                          "boş" ile "dolu" karışmasın)
//    İndeksler sona gelince başa sarar (& mask_, çünkü Capacity 2'nin kuvveti).
//
//  BELLEK SIRALAMASI (memory ordering) — lock-free'nin kalbi:
//    - Üretici: önce veriyi buffer_'a yazar, SONRA head_'i `release` ile ilerletir.
//    - Tüketici: head_'i `acquire` ile okur; bu, üreticinin veri yazımının
//      görünür olmasını GARANTİ eder. (release-store ↔ acquire-load eşleşmesi)
//    Yanlış sıralama → "nadiren tekrarlanan" race hataları. Bu yüzden dikkat.
// ----------------------------------------------------------------------------
#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

namespace dtrack {

template <typename T, std::size_t Capacity>
class SpscRingBuffer {
    static_assert(Capacity >= 2, "Kapasite >= 2 olmali");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Kapasite 2'nin kuvveti olmali (hizli sarma: & mask)");

public:
    SpscRingBuffer() = default;

    // Kopyalama/taşıma kapalı: atomic sayaçlar taşınamaz, ayrıca paylaşılan
    // bir kuyruğun kopyalanması mantıksız.
    SpscRingBuffer(const SpscRingBuffer&)            = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    // ----- ÜRETİCİ tarafı: push -----
    // Öğeyi kuyruğa ekler. Kuyruk doluysa false döner (öğe eklenmez).
    bool push(const T& item) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & mask_;
        // Tüketicinin nerede olduğunu acquire ile oku (onun pop'u görünür olsun).
        if (next == tail_.load(std::memory_order_acquire)) {
            return false; // DOLU
        }
        buffer_[head] = item;
        // Veri yazımını yayınla: head_'i release ile ilerlet.
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Taşıma (move) sürümü — kopya yerine taşır (örn. cv::Mat için ucuz).
    bool push(T&& item) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & mask_;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false; // DOLU
        }
        buffer_[head] = std::move(item);
        head_.store(next, std::memory_order_release);
        return true;
    }

    // ----- TÜKETİCİ tarafı: pop -----
    // Sıradaki öğeyi 'out'a alır. Kuyruk boşsa false döner.
    bool pop(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        // Üreticinin nerede olduğunu acquire ile oku (onun verisi görünür olsun).
        if (tail == head_.load(std::memory_order_acquire)) {
            return false; // BOŞ
        }
        out = std::move(buffer_[tail]);
        // Slotu serbest bıraktığımızı yayınla.
        tail_.store((tail + 1) & mask_, std::memory_order_release);
        return true;
    }

    // Yardımcılar (yaklaşık; eşzamanlı erişimde anlık fotoğraf).
    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    // Kullanılabilir maksimum öğe sayısı (1 slot ayrıldığı için Capacity-1).
    static constexpr std::size_t capacity() { return Capacity - 1; }

private:
    static constexpr std::size_t mask_ = Capacity - 1;

    std::array<T, Capacity> buffer_{};

    // alignas(64): head_ ve tail_'i ayrı cache satırlarına koyar.
    // ÖĞREN: "false sharing" = iki thread'in farklı değişkenleri aynı cache
    // satırına düşünce CPU gereksiz yere senkronize eder ve yavaşlar. Ayırınca
    // üretici head_'i, tüketici tail_'i bağımsızca günceller.
    alignas(64) std::atomic<std::size_t> head_{0}; // sadece ÜRETİCİ yazar
    alignas(64) std::atomic<std::size_t> tail_{0}; // sadece TÜKETİCİ yazar
};

} // namespace dtrack
