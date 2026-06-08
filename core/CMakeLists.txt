# core/ — paylaşılan tipler + lock-free ring buffer (sadece header).
#
# ÖĞREN: "INTERFACE library" = derlenecek .cpp'si olmayan, sadece header
# sağlayan hedef. Ona bağlanan herkes include yollarını ve bağımlılıkları
# (burada OpenCV) otomatik miras alır.

add_library(dtrack_core INTERFACE)

target_include_directories(dtrack_core INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(dtrack_core INTERFACE ${OpenCV_LIBS})
