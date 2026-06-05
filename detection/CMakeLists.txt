# dtrack_detection: top-hat + MOG2 küçük hedef tespiti.
add_library(dtrack_detection STATIC
    src/mog_detector.cpp
    src/stability_discriminator.cpp)

target_include_directories(dtrack_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_detection PUBLIC
    dtrack_common
    dtrack_pipeline
    ${OpenCV_LIBS})
target_include_directories(dtrack_detection PUBLIC ${OpenCV_INCLUDE_DIRS})
