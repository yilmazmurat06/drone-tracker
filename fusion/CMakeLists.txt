# dtrack_fusion: track-seviyesi gec fuzyon (Problem 5).
add_library(dtrack_fusion STATIC
    src/simple_track_fusion.cpp)

target_include_directories(dtrack_fusion PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_fusion PUBLIC
    dtrack_common
    ${OpenCV_LIBS})
target_include_directories(dtrack_fusion PUBLIC ${OpenCV_INCLUDE_DIRS})
