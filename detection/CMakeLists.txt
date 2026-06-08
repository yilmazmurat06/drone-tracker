# detection/ — P2 tespit + P3 ayraç. Artık .cpp içeriyor → STATIC kütüphane.
add_library(dtrack_detection STATIC
  src/moving_target_detector.cpp
  src/clutter_discriminator.cpp)
target_include_directories(dtrack_detection PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_detection PUBLIC dtrack_core)
