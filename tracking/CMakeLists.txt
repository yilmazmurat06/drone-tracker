# dtrack_tracking: çoklu-hedef takip + M-of-N yaşam döngüsü.
#   - kalman_tracker.cpp     : GERÇEK 4-durumlu CV Kalman (varsayılan); kalman_core.hpp (Kf2)
#   - alpha_beta_tracker.cpp : α-β (Kalata) sabit-kazanç filtresi (yedek/karşılaştırma)
#   - imm_tracker.cpp        : iki Kalman modelli IMM (yumuşak + çevik)
# Çekirdek matematik (Kf2) kalman_core.hpp'de, OpenCV'siz (tests/test_kalman_math.cpp ile sınanır).
add_library(dtrack_tracking STATIC
    src/kalman_tracker.cpp
    src/alpha_beta_tracker.cpp
    src/imm_tracker.cpp)

target_include_directories(dtrack_tracking PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(dtrack_tracking PUBLIC
    dtrack_common
    dtrack_pipeline
    ${OpenCV_LIBS})
target_include_directories(dtrack_tracking PUBLIC ${OpenCV_INCLUDE_DIRS})
