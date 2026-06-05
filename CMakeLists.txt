cmake_minimum_required(VERSION 3.16)
project(drone_tracker LANGUAGES CXX)

# --- Dil standardı -----------------------------------------------------------
# C++20: nested namespace, std::optional, designated initializers vb. kullanıyoruz.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# IDE/clangd için compile_commands.json üret (header'lardaki kırmızı çizgiler kalkar).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ARM hedefte sıcak döngüler için agresif optimizasyon; native debug'da kapalı.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(-Wall -Wextra)

# --- Bağımlılıklar -----------------------------------------------------------
find_package(Threads REQUIRED)  # pipeline thread'leri için (pthread)

# OpenCV opsiyonel: bulunamazsa çekirdek (common + pipeline + testler) yine derlenir.
# Bileşenler: core (matris), imgproc (warp/blur/köşe), imgcodecs (PNG yaz/oku),
# videoio (VideoCapture -> simülatör/dosya/donanım), video (MOG2 + LK optical flow),
# calib3d (estimateAffinePartial2D -> RANSAC ego-motion kestirimi).
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio video calib3d highgui)

option(DTRACK_BUILD_TESTS "Birim testlerini derle" ON)

# --- Modüller ----------------------------------------------------------------
# Çekirdek her zaman derlenir (OpenCV'siz):
add_subdirectory(common)
add_subdirectory(pipeline)

if(OpenCV_FOUND)
    message(STATUS "OpenCV bulundu (${OpenCV_VERSION}) -> tüm modüller derlenecek.")
    add_subdirectory(io)
    add_subdirectory(stabilization)
    add_subdirectory(detection)
    add_subdirectory(tracking)
    add_subdirectory(fusion)
    add_subdirectory(app)
else()
    message(WARNING
        "OpenCV bulunamadi. Yalnizca cekirdek (common + pipeline) ve testler derlenecek.\n"
        "  macOS:  brew install opencv\n"
        "  Ubuntu: sudo apt install libopencv-dev")
endif()

if(DTRACK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
