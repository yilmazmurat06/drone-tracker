# stabilization/ — P1: ego-motion stabilizasyon.
# Artık .cpp içeriyor → STATIC kütüphane.
add_library(dtrack_stabilization STATIC
  src/optical_flow_stabilizer.cpp
  src/gyro_flow_stabilizer.cpp)
target_include_directories(dtrack_stabilization PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(dtrack_stabilization PUBLIC dtrack_core)
