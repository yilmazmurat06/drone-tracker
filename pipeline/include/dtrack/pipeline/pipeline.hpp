#pragma once
//
// Pipeline: tüm stage'lerin sahibi. Hepsini doğru sırada başlatır/durdurur.
//
// Stage'ler kuyruklarla zaten birbirine bağlanmış olarak gelir (connect()).
// Pipeline yalnızca yaşam döngüsünü (lifecycle) yönetir:
//   - Tüketiciler önce başlar, üreticiler sonra  -> ilk veride kuyruk hazır.
//   - Üreticiler önce durur, tüketiciler sonra    -> veri sızıntısı/asılma olmaz.
//
// IStage: Stage<In,Out> şablonunu tip-bağımsız tutmak için minimal arayüz.
// (Pipeline farklı In/Out tiplerindeki stage'leri tek listede tutabilsin diye.)

#include <memory>
#include <vector>

#include "dtrack/pipeline/istage.hpp"

namespace dtrack::pipeline {

class Pipeline {
public:
    // Stage'leri AKIŞ SIRASINA göre ekle (kaynak -> ... -> sink).
    void add(std::shared_ptr<IStage> stage) { stages_.push_back(std::move(stage)); }

    // Tüketiciden üreticiye doğru başlat (ters sıra).
    void start() {
        for (auto it = stages_.rbegin(); it != stages_.rend(); ++it) {
            (*it)->start();
        }
    }

    // Üreticiden tüketiciye doğru durdur (düz sıra).
    void stop() {
        for (auto& s : stages_) s->stop();
    }

private:
    std::vector<std::shared_ptr<IStage>> stages_;
};

}  // namespace dtrack::pipeline
