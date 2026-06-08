# tracking/ — P4: Kalman takip. Artık .cpp içeriyor → STATIC kütüphane.
add_library(dtrack_tracking STATIC
  src/multi_target_tracker.cpp)
target_include_directories(dtrack_tracking PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_tracking PUBLIC dtrack_core)
