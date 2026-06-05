# dtrack_stabilization: ego-motion telafisi (gyro + optical flow füzyonu).
add_library(dtrack_stabilization STATIC
    src/klt_gyro_stabilizer.cpp)

target_include_directories(dtrack_stabilization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_stabilization PUBLIC
    dtrack_common
    dtrack_pipeline
    dtrack_io          # StabilizeStage IImuSource'u kullanır
    ${OpenCV_LIBS})
target_include_directories(dtrack_stabilization PUBLIC ${OpenCV_INCLUDE_DIRS})
