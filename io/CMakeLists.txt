# io/ — veri kaynakları.
#
# ÖĞREN: Artık .cpp'lerimiz var, bu yüzden INTERFACE değil DERLENEN (STATIC)
# kütüphane. INTERFACE = sadece header; STATIC = derlenip .a'ya konan gerçek kod.
# PUBLIC bağımlılık → dtrack_io'ya bağlanan (örn. app) hem header yollarını hem
# OpenCV'yi (dtrack_core üzerinden) miras alır.

add_library(dtrack_io STATIC
  src/telemetry_log.cpp
  src/recorded_frame_source.cpp
  src/recorded_telemetry_source.cpp)

target_include_directories(dtrack_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_io PUBLIC dtrack_core)
