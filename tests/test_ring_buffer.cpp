// SPSC lock-free ring buffer testleri.
// Harici test framework'u yok (gomulu hedefte bagimlilik az olsun) -> minik bir
// assert makrosu kullaniyoruz. Basarisizlikta non-zero exit -> ctest "FAILED" der.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include "dtrack/common/ring_buffer.hpp"

using dtrack::common::SpscRingBuffer;

static int g_failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL: %s (satir %d)\n", #cond, __LINE__);      \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

static void test_basic_push_pop() {
    std::printf("test_basic_push_pop\n");
    SpscRingBuffer<int> q(4);
    CHECK(q.empty());
    CHECK(q.push(1));
    CHECK(q.push(2));
    CHECK(!q.empty());
    CHECK(q.size_approx() == 2);
    auto a = q.pop();
    CHECK(a.has_value() && *a == 1);
    auto b = q.pop();
    CHECK(b.has_value() && *b == 2);
    CHECK(!q.pop().has_value());  // bos
    CHECK(q.empty());
}

static void test_full_rejects() {
    std::printf("test_full_rejects\n");
    SpscRingBuffer<int> q(2);  // en fazla 2 eleman
    CHECK(q.push(10));
    CHECK(q.push(20));
    CHECK(!q.push(30));  // dolu -> reddet
    CHECK(q.size_approx() == 2);
}

static void test_overwrite_drops_oldest() {
    std::printf("test_overwrite_drops_oldest\n");
    SpscRingBuffer<int> q(2);
    q.push_overwrite(1);
    q.push_overwrite(2);
    int discarded = -1;
    q.push_overwrite(3, &discarded);  // dolu -> en eski (1) dusurulur
    CHECK(discarded == 1);
    auto x = q.pop();
    auto y = q.pop();
    CHECK(x.has_value() && *x == 2);
    CHECK(y.has_value() && *y == 3);
}

// En kritik test: gercek iki-thread'li uretici/tuketici altinda hicbir eleman
// kaybolmamali, bozulmamali ve sira korunmali.
static void test_spsc_threaded() {
    std::printf("test_spsc_threaded\n");
    constexpr int N = 1'000'000;
    SpscRingBuffer<int> q(1024);

    std::thread producer([&] {
        for (int i = 0; i < N; ++i) {
            while (!q.push(i)) {
                std::this_thread::yield();  // dolu -> tuketiciye yer ac
            }
        }
    });

    long long sum = 0;
    int count = 0;
    std::thread consumer([&] {
        int expected = 0;
        while (count < N) {
            auto v = q.pop();
            if (!v) {
                std::this_thread::yield();
                continue;
            }
            if (*v != expected) {  // sira bozulmus -> SPSC garantisi kirik
                std::printf("  FAIL: sira hatasi, beklenen %d geldi %d\n", expected, *v);
                ++g_failures;
            }
            ++expected;
            sum += *v;
            ++count;
        }
    });

    producer.join();
    consumer.join();

    CHECK(count == N);
    // 0..N-1 toplami = N*(N-1)/2  -> hicbir eleman kaybolmadi/tekrarlanmadi.
    const long long expected_sum = static_cast<long long>(N) * (N - 1) / 2;
    CHECK(sum == expected_sum);
}

int main() {
    test_basic_push_pop();
    test_full_rejects();
    test_overwrite_drops_oldest();
    test_spsc_threaded();

    if (g_failures == 0) {
        std::printf("TUM TESTLER GECTI\n");
        return 0;
    }
    std::printf("%d TEST BASARISIZ\n", g_failures);
    return 1;
}
