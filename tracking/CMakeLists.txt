# dtrack_tracking: çoklu-hedef Kalman (sabit-ivme) takip + M-of-N yaşam döngüsü.
add_library(dtrack_tracking STATIC
    src/kalman_tracker.cpp)

target_include_directories(dtrack_tracking PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_tracking PUBLIC
    dtrack_common
    dtrack_pipeline
    ${OpenCV_LIBS})
target_include_directories(dtrack_tracking PUBLIC ${OpenCV_INCLUDE_DIRS})
