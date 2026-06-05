# dtrack_detection: küçük hedef tespiti.
#   - mog_detector.cpp        : top-hat + MOG2 (gökyüzü, 2-6 px parlak nokta)
#   - sky_region_detector.cpp : gökyüzü-kapılı hareket (arazi-üstü, belirgin hedef)
#   - stability_discriminator.cpp : drone skorlama
add_library(dtrack_detection STATIC
    src/mog_detector.cpp
    src/sky_region_detector.cpp
    src/stability_discriminator.cpp)

target_include_directories(dtrack_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_detection PUBLIC
    dtrack_common
    dtrack_pipeline
    ${OpenCV_LIBS})
target_include_directories(dtrack_detection PUBLIC ${OpenCV_INCLUDE_DIRS})
