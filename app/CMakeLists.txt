# drone-tracker uygulamasi (iskelet). Yalnizca OpenCV bulundugunda derlenir
# (kok CMakeLists bu add_subdirectory'i OpenCV_FOUND altinda cagirir).
add_executable(drone_tracker_app main.cpp)

set_target_properties(drone_tracker_app PROPERTIES OUTPUT_NAME drone_tracker)

target_link_libraries(drone_tracker_app PRIVATE
    dtrack_fusion
    dtrack_tracking
    dtrack_detection
    dtrack_stabilization
    dtrack_io
    dtrack_pipeline
    dtrack_common)

# Video benchmark aracı: MP4 dosyalarını pipeline'dan geçirir.
add_executable(video_bench video_bench.cpp)
set_target_properties(video_bench PROPERTIES OUTPUT_NAME video_bench)

target_link_libraries(video_bench PRIVATE
    dtrack_tracking
    dtrack_detection
    dtrack_stabilization
    dtrack_io
    dtrack_common
    ${OpenCV_LIBS})
