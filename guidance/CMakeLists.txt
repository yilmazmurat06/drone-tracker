# guidance/ — Output/güdüm katmanı: tek-hedef tracker arayüzü + durum makinesi.
# Tracker NPU-hedefli (şimdilik NCC CPU stub); detector/discriminator CPU'da kalır.
add_library(dtrack_guidance STATIC
  src/ncc_template_tracker.cpp
  src/nano_siamese_tracker.cpp
  src/guidance_controller.cpp)
target_include_directories(dtrack_guidance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_guidance PUBLIC
  dtrack_core
  dtrack_detection)
