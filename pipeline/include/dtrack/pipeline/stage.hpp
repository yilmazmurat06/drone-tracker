#pragma once
//
// Stage: pipeline'daki bir işlem aşamasının ortak thread iskeleti.
//
// Her stage kendi thread'inde döner:
//   girdi kuyruğundan al  ->  process()  ->  çıktı kuyruğuna koy
//
// Somut bir aşama yazmak için tek yapman gereken process()'i doldurmak.
// Thread başlatma/durdurma, kuyruk yönetimi ve geriye-basınç (back-pressure)
// politikası burada bir kez halledilir (bkz. Problem 6).
//
// Şablon parametreleri:
//   In  = girdi eleman tipi (örn. FramePtr)        -> kaynak stage'lerde void
//   Out = çıktı eleman tipi (örn. StabilizedFrame) -> uç (sink) stage'lerde void

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "dtrack/common/ring_buffer.hpp"
#include "dtrack/pipeline/istage.hpp"

namespace dtrack::pipeline {

template <typename In, typename Out>
class Stage : public IStage {
public:
    using InQueue = common::SpscRingBuffer<In>;
    using OutQueue = common::SpscRingBuffer<Out>;

    explicit Stage(std::string name) : name_(std::move(name)) {}
    virtual ~Stage() { stop(); }

    // Kuyrukları bağla. input kaynak stage'de nullptr, output sink stage'de nullptr.
    void connect(std::shared_ptr<InQueue> input, std::shared_ptr<OutQueue> output) {
        input_ = std::move(input);
        output_ = std::move(output);
    }

    void start() override {
        if (running_.exchange(true)) return;  // zaten çalışıyor
        thread_ = std::thread([this] { run_loop(); });
    }

    void stop() override {
        if (!running_.exchange(false)) return;
        if (thread_.joinable()) thread_.join();
    }

    const std::string& name() const override { return name_; }

    // Basit metrikler (debug / latency profili için).
    std::uint64_t processed() const { return processed_.load(std::memory_order_relaxed); }
    std::uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

protected:
    // Asıl iş burada. Bir girdiyi alıp isteğe bağlı bir çıktı üretir.
    // std::nullopt döndürmek "bu girdi çıktı üretmedi" demektir (örn. tespit yok).
    virtual std::optional<Out> process(In&& input) = 0;

    // Çıktıyı bir sonraki stage'e gönderir. drop-oldest politikasıyla:
    // kuyruk doluysa en eski çıktı düşürülür ki tüketici hep en taze veriyi görsün.
    void emit(Out&& out) {
        if (!output_) return;
        Out discarded;
        const std::size_t before = output_->size_approx();
        output_->push_overwrite(std::move(out), &discarded);
        if (output_->size_approx() <= before && before == output_->capacity()) {
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
    }

private:
    void run_loop() {
        while (running_.load(std::memory_order_relaxed)) {
            if (!input_) {
                // Kaynak stage (kamera/IMU): girdisi yok, kendi verisini üretir.
                // Üretimi process()'e bırakırız; boş bir girdi geçeriz.
                auto out = process(In{});
                if (out) emit(std::move(*out));
                processed_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            auto item = input_->pop();
            if (!item) {
                // Girdi yok: kısa uyku ile CPU yakmadan bekle (busy-spin değil).
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            auto out = process(std::move(*item));
            processed_.fetch_add(1, std::memory_order_relaxed);
            if (out) emit(std::move(*out));
        }
    }

    std::string name_;
    std::shared_ptr<InQueue> input_;
    std::shared_ptr<OutQueue> output_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<std::uint64_t> processed_{0};
    std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace dtrack::pipeline
