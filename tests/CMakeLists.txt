# tests/ — birim testleri (GoogleTest).
#
# ÖĞREN: gtest_discover_tests, çalıştırılabilirdeki her TEST(...) bloğunu
# otomatik bulup ctest'e kaydeder; "ctest" hepsini koşar.

include(GoogleTest)

add_executable(test_ring_buffer test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE
  dtrack_core
  GTest::gtest_main
  Threads::Threads)
gtest_discover_tests(test_ring_buffer)

add_executable(test_telemetry_interp test_telemetry_interp.cpp)
target_link_libraries(test_telemetry_interp PRIVATE
  dtrack_io
  GTest::gtest_main)
gtest_discover_tests(test_telemetry_interp)

add_executable(test_stabilizer test_stabilizer.cpp)
target_link_libraries(test_stabilizer PRIVATE
  dtrack_stabilization
  GTest::gtest_main)
gtest_discover_tests(test_stabilizer)

add_executable(test_gyro_stabilizer test_gyro_stabilizer.cpp)
target_link_libraries(test_gyro_stabilizer PRIVATE
  dtrack_stabilization
  GTest::gtest_main)
gtest_discover_tests(test_gyro_stabilizer)

add_executable(test_detector test_detector.cpp)
target_link_libraries(test_detector PRIVATE
  dtrack_detection
  GTest::gtest_main)
gtest_discover_tests(test_detector)

add_executable(test_tracker test_tracker.cpp)
target_link_libraries(test_tracker PRIVATE
  dtrack_tracking
  GTest::gtest_main)
gtest_discover_tests(test_tracker)

add_executable(test_discriminator test_discriminator.cpp)
target_link_libraries(test_discriminator PRIVATE
  dtrack_detection
  GTest::gtest_main)
gtest_discover_tests(test_discriminator)

add_executable(test_guidance test_guidance.cpp)
target_link_libraries(test_guidance PRIVATE
  dtrack_guidance
  GTest::gtest_main)
gtest_discover_tests(test_guidance)

add_executable(test_lock_integrity test_lock_integrity.cpp)
target_link_libraries(test_lock_integrity PRIVATE
  dtrack_guidance
  GTest::gtest_main)
gtest_discover_tests(test_lock_integrity)
