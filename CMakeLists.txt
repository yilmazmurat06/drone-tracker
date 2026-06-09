# ============================================================================
#  drone-tracker — CPU-only hava-hava drone tespit & takip sistemi
#  Üst seviye CMake yapılandırması (Adım 1: iskelet)
# ============================================================================
#
#  ÖĞREN: CMake bir "build sistemi üreticisi"dir. Bu dosya NE derleneceğini
#  ve bağımlılıkları tanımlar; CMake bundan Makefile/Ninja üretir.
#  Tipik kullanım:
#      cmake -B build            # yapılandır (build/ klasörü üretir)
#      cmake --build build       # derle
#      ctest --test-dir build    # testleri çalıştır
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

project(drone_tracker
        VERSION 0.1.0
        DESCRIPTION "CPU-only air-to-air drone detection & tracking"
        LANGUAGES CXX)

# --- C++ standardı: C++17, derleyiciye özel uzantılar kapalı (taşınabilirlik) ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Varsayılan build tipi: Release (optimizasyon açık). Gerçek-zamanlı hedef. ---
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(DTRACK_BUILD_TESTS "Birim testlerini derle (GoogleTest)" ON)

# --- Bağımlılıklar ---
# OpenCV: görüntü işleme (cv::Mat, VideoCapture, MOG2, optical flow ...).
find_package(OpenCV REQUIRED)
# Threads: std::thread / std::atomic için (ring buffer testi, ileride pipeline).
find_package(Threads REQUIRED)

message(STATUS "OpenCV bulundu: ${OpenCV_VERSION}  (include: ${OpenCV_INCLUDE_DIRS})")

# --- Modüller (her biri kendi CMakeLists.txt'ine sahip) ---
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(stabilization)
add_subdirectory(detection)
add_subdirectory(tracking)
add_subdirectory(guidance)
add_subdirectory(pipeline)
add_subdirectory(app)

# --- Testler: GoogleTest'i FetchContent ile internetten çek ---
if(DTRACK_BUILD_TESTS)
  enable_testing()
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  # Windows'ta CRT uyumu için (zararsız, diğer platformda yok sayılır)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)
endif()
