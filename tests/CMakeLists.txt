# Ring buffer testi OpenCV'siz derlenir (cekirdegi hemen dogrulayabilirsin).
add_executable(test_ring_buffer test_ring_buffer.cpp)
target_link_libraries(test_ring_buffer PRIVATE
    dtrack_common
    Threads::Threads)
add_test(NAME ring_buffer COMMAND test_ring_buffer)

# Kalman cekirdegi (Kf2) saf matematik testi: OpenCV'siz derlenir/kosar.
# Kalman tracker'in kullandigi kalman_core.hpp'yi dogrudan icerir; §2.7'de
# Kalman'in terk edilmesine yol acan sorunlarin (P kararsizligi, salinim)
# giderildigini kanitlar. Header-only -> sadece include yolu yeterli.
add_executable(test_kalman_math test_kalman_math.cpp)
target_include_directories(test_kalman_math PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../tracking/include)
add_test(NAME kalman_math COMMAND test_kalman_math)

# Sentetik kaynak testi OpenCV + dtrack_io gerektirir -> yalniz OpenCV varsa.
if(OpenCV_FOUND)
    add_executable(test_synthetic_source test_synthetic_source.cpp)
    target_link_libraries(test_synthetic_source PRIVATE dtrack_io)
    add_test(NAME synthetic_source COMMAND test_synthetic_source)

    add_executable(test_stabilizer test_stabilizer.cpp)
    target_link_libraries(test_stabilizer PRIVATE dtrack_stabilization dtrack_io)
    add_test(NAME stabilizer COMMAND test_stabilizer)

    add_executable(test_detector test_detector.cpp)
    target_link_libraries(test_detector PRIVATE
        dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME detector COMMAND test_detector)

    # Birincil takip testi: KalmanTracker kilit surekliligi/dogrulugu/ID kararliligi.
    add_executable(test_tracker test_tracker.cpp)
    target_link_libraries(test_tracker PRIVATE
        dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME tracker COMMAND test_tracker)

    # Yedek takip testi: AlphaBetaTracker ayni cita (Kalman ile karsilastirma temeli).
    add_executable(test_alpha_beta_tracker test_alpha_beta_tracker.cpp)
    target_link_libraries(test_alpha_beta_tracker PRIVATE
        dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME alpha_beta_tracker COMMAND test_alpha_beta_tracker)

    add_executable(test_discriminator test_discriminator.cpp)
    target_link_libraries(test_discriminator PRIVATE
        dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME discriminator COMMAND test_discriminator)

    add_executable(test_fusion test_fusion.cpp)
    target_link_libraries(test_fusion PRIVATE
        dtrack_fusion dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME fusion COMMAND test_fusion)

    # Track hiz (velocity) cikti dogrulugu + fps degismezligi regresyon testi.
    add_executable(test_track_velocity test_track_velocity.cpp)
    target_link_libraries(test_track_velocity PRIVATE
        dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME track_velocity COMMAND test_track_velocity)

    # IMM tracker (iki Kalman modelli): tek Kalman ile ayni saglamlik citasini tutturmali.
    add_executable(test_imm_tracker test_imm_tracker.cpp)
    target_link_libraries(test_imm_tracker PRIVATE
        dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
    add_test(NAME imm_tracker COMMAND test_imm_tracker)

    add_executable(bench_pipeline bench_pipeline.cpp)
    target_link_libraries(bench_pipeline PRIVATE
        dtrack_tracking dtrack_detection dtrack_stabilization dtrack_io)
endif()
