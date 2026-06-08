// ============================================================================
//  dtrack_app — iskelet doğrulama programı.
//  Henüz bir şey "takip" etmez; sadece build'in ayakta ve bağımlılıkların
//  (OpenCV, threads, çekirdek tipler) doğru bağlandığını gösterir.
// ============================================================================
#include <iostream>
#include <thread>

#include <opencv2/core.hpp>

#include "dtrack/core/time.hpp"
#include "dtrack/core/ring_buffer.hpp"

int main() {
    std::cout << "==== drone-tracker (iskelet, Adim 1) ====\n";
    std::cout << "OpenCV surumu       : " << CV_VERSION << "\n";
    std::cout << "Donanim thread'leri : " << std::thread::hardware_concurrency() << "\n";
    std::cout << "now_ns()            : " << dtrack::now_ns() << "\n";

    // Ring buffer'ın gerçekten linklendiğini/çalıştığını minik bir kullanımla göster.
    dtrack::SpscRingBuffer<int, 8> rb;
    rb.push(1);
    rb.push(2);
    int a = 0, b = 0;
    rb.pop(a);
    rb.pop(b);
    std::cout << "ring buffer demo    : " << a << ", " << b << " (beklenen 1, 2)\n";

    std::cout << "Iskelet ayakta. Sonraki adim: io/ replay (kayitli .mp4 + .csv).\n";
    return 0;
}
