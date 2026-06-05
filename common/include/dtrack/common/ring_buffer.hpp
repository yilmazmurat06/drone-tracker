#pragma once
//
// SPSC (Single Producer, Single Consumer) lock-free ring buffer.
//
// "Lock-free" = thread'ler bir kilit (mutex) için beklemez; senkronizasyon
// yalnızca atomic değişkenler ve bellek sıralaması (memory ordering) ile yapılır.
//
// Bu sınıf SADECE tek üretici + tek tüketici durumu için güvenlidir. Pipeline'da
// her kuyruk tam olarak iki stage'i bağladığı için bu varsayım her zaman geçerli.
// İki tarafın aynı anda farklı uçlara dokunması, daha karmaşık (ve yavaş) genel
// kuyrukları gereksiz kılar.
//
// Tasarım: kapasiteden bir fazla slot ayırırız (head == tail "boş", head'in tail'i
// "yakalaması" "dolu" demektir). Böylece dolu/boş ayrımı için ekstra sayaç gerekmez.

#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace dtrack::common {

template <typename T>
class SpscRingBuffer {
public:
    // capacity = aynı anda tutulabilecek en fazla eleman sayısı.
    // İçeride capacity+1 slot ayrılır (bir slot dolu/boş ayrımı için kullanılır).
    explicit SpscRingBuffer(std::size_t capacity)
        : slots_(capacity + 1), capacity_(capacity) {}

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

    // Üretici tarafından çağrılır. Kuyruk doluysa false döner (eleman atılmaz).
    bool push(T value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = increment(head);
        // acquire: tüketicinin tail güncellemesini görelim ki "dolu mu?" doğru olsun.
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // dolu
        }
        slots_[head] = std::move(value);
        // release: slot yazımı, head görünür olmadan ÖNCE tamamlansın.
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Üretici tarafından çağrılır. Kuyruk doluysa EN ESKİ elemanı düşürür ve
    // yenisini yazar. Gerçek-zamanlı "her zaman en taze veriyi tut" politikası için.
    // discarded != nullptr ise düşürülen eleman oraya taşınır.
    void push_overwrite(T value, T* discarded = nullptr) {
        while (!push(value)) {
            T old;
            if (pop_into(old) && discarded != nullptr) {
                *discarded = std::move(old);
            }
        }
    }

    // Tüketici tarafından çağrılır. Boşsa std::nullopt döner.
    std::optional<T> pop() {
        T out;
        if (pop_into(out)) {
            return std::optional<T>(std::move(out));
        }
        return std::nullopt;
    }

    bool empty() const {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    std::size_t capacity() const { return capacity_; }

    // Anlık yaklaşık eleman sayısı (debug/metrik için; tam senkron değildir).
    std::size_t size_approx() const {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h + slots_.size() - t) % slots_.size();
    }

private:
    bool pop_into(T& out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // boş
        }
        out = std::move(slots_[tail]);
        tail_.store(increment(tail), std::memory_order_release);
        return true;
    }

    std::size_t increment(std::size_t i) const { return (i + 1) % slots_.size(); }

    std::vector<T> slots_;
    const std::size_t capacity_;
    // head: üreticinin yazacağı slot. tail: tüketicinin okuyacağı slot.
    // Farklı cache satırlarında olmaları için ileride alignas eklenebilir.
    std::atomic<std::size_t> head_{0};
    std::atomic<std::size_t> tail_{0};
};

}  // namespace dtrack::common
