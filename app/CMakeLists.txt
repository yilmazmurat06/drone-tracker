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

# Headless degerlendirme araci: gercek video + (ops.) GT CSV -> bench metrikleri.
add_executable(video_eval video_eval.cpp)
set_target_properties(video_eval PROPERTIES OUTPUT_NAME video_eval)
target_link_libraries(video_eval PRIVATE
    dtrack_tracking
    dtrack_detection
    dtrack_stabilization
    dtrack_io
    dtrack_common
    ${OpenCV_LIBS})

# live_track: etkilesimli "cued" takip (fare ile hedef sec -> CSRT/KCF kilitlenir).
# CSRT/KCF contrib 'tracking' modulundedir; yoksa bu hedef atlanir (cekirdek etkilenmez).
find_package(OpenCV QUIET COMPONENTS tracking)
if(TARGET opencv_tracking)
    add_executable(live_track live_track.cpp)
    set_target_properties(live_track PROPERTIES OUTPUT_NAME live_track)
    target_link_libraries(live_track PRIVATE
        dtrack_common
        opencv_tracking
        opencv_highgui
        opencv_videoio
        opencv_imgproc
        opencv_core)
else()
    message(STATUS "opencv_tracking modulu yok -> live_track atlandi "
                   "(CSRT/KCF icin OpenCV contrib gerekir).")
endif()

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
