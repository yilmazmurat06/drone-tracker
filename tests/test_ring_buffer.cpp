// ============================================================================
//  SpscRingBuffer birim testleri.
//  Lock-free kod "çoğu zaman" çalışıp nadiren bozulabilir; bu yüzden hem temel
//  davranışı hem de gerçek iki-thread'li senaryoyu test ediyoruz.
// ============================================================================
#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "dtrack/core/ring_buffer.hpp"

using dtrack::SpscRingBuffer;

// Boş kuyruktan pop başarısız olmalı.
TEST(RingBuffer, EmptyPopFails) {
    SpscRingBuffer<int, 4> rb;
    int x = -1;
    EXPECT_FALSE(rb.pop(x));
    EXPECT_TRUE(rb.empty());
}

// Tek push → tek pop, aynı değeri verir.
TEST(RingBuffer, SinglePushPop) {
    SpscRingBuffer<int, 4> rb;
    EXPECT_TRUE(rb.push(42));
    EXPECT_FALSE(rb.empty());

    int x = 0;
    EXPECT_TRUE(rb.pop(x));
    EXPECT_EQ(x, 42);
    EXPECT_TRUE(rb.empty());
}

// Kapasite (Capacity-1) dolunca push başarısız olmalı.
TEST(RingBuffer, FillToCapacityThenFull) {
    SpscRingBuffer<int, 4> rb;  // kullanılabilir kapasite = 3
    EXPECT_EQ(rb.capacity(), 3u);
    EXPECT_TRUE(rb.push(1));
    EXPECT_TRUE(rb.push(2));
    EXPECT_TRUE(rb.push(3));
    EXPECT_FALSE(rb.push(4));   // DOLU
}

// FIFO sırası korunur.
TEST(RingBuffer, FifoOrder) {
    SpscRingBuffer<int, 8> rb;
    for (int i = 0; i < 5; ++i) ASSERT_TRUE(rb.push(i));
    for (int i = 0; i < 5; ++i) {
        int x = -1;
        ASSERT_TRUE(rb.pop(x));
        EXPECT_EQ(x, i);
    }
    EXPECT_TRUE(rb.empty());
}

// Sarma (wrap-around): defalarca dolup boşalınca indeks başa sarmalı.
TEST(RingBuffer, WrapAround) {
    SpscRingBuffer<int, 4> rb;  // kapasite 3
    for (int round = 0; round < 1000; ++round) {
        ASSERT_TRUE(rb.push(round));
        int x = -1;
        ASSERT_TRUE(rb.pop(x));
        EXPECT_EQ(x, round);
    }
}

// Asıl sınav: bir ÜRETİCİ + bir TÜKETİCİ thread, kayıp/sıra bozulması olmadan.
TEST(RingBuffer, SpscThreadedNoLoss) {
    constexpr int N = 100000;
    SpscRingBuffer<int, 1024> rb;

    std::vector<int> received;
    received.reserve(N);

    std::thread producer([&] {
        for (int i = 0; i < N; /* push başarılıysa ilerle */) {
            if (rb.push(i)) {
                ++i;
            } else {
                std::this_thread::yield();  // kuyruk dolu → tüketiciye sıra ver
            }
        }
    });

    std::thread consumer([&] {
        int got = 0;
        int x = 0;
        while (got < N) {
            if (rb.pop(x)) {
                received.push_back(x);
                ++got;
            } else {
                std::this_thread::yield();  // kuyruk boş → üreticiye sıra ver
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(static_cast<int>(received.size()), N);
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(received[i], i);  // hiçbir öğe kaybolmadı ve sıra korundu
    }
}
