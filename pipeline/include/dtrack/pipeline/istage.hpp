#pragma once
//
// IStage: bir stage'in tip-bağımsız (In/Out şablonundan arınmış) minimal arayüzü.
// Pipeline, farklı girdi/çıktı tiplerindeki stage'leri tek bir listede tutabilsin
// ve hepsini ortak şekilde start()/stop() edebilsin diye var.

#include <string>

namespace dtrack::pipeline {

class IStage {
public:
    virtual ~IStage() = default;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual const std::string& name() const = 0;
};

}  // namespace dtrack::pipeline
