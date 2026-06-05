# dtrack_io: kamera & IMU kaynakları (sentetik + video/simülatör + donanım kapısı).
# Artık .cpp implementasyonları var -> STATIC kütüphane.
add_library(dtrack_io STATIC
    src/synthetic_camera_source.cpp
    src/synthetic_imu_source.cpp
    src/video_camera_source.cpp)

target_include_directories(dtrack_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# PUBLIC: io başlıklarını kullanan herkes bu bağımlılıkları da görür.
target_link_libraries(dtrack_io PUBLIC
    dtrack_common
    dtrack_pipeline
    ${OpenCV_LIBS})
target_include_directories(dtrack_io PUBLIC ${OpenCV_INCLUDE_DIRS})
