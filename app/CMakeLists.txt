# app/ — çalıştırılabilir giriş noktaları.

# İskelet doğrulama programı. (dtrack_io, dtrack_pipeline üzerinden geliyor.)
add_executable(dtrack_app main.cpp)
target_link_libraries(dtrack_app PRIVATE
  dtrack_core
  dtrack_pipeline
  Threads::Threads)

# Replay oynatıcı (Adım 2): mp4 + csv telemetri overlay.
add_executable(dtrack_replay replay.cpp)
target_link_libraries(dtrack_replay PRIVATE
  dtrack_core
  dtrack_io
  Threads::Threads)

# Stabilizasyon demo (Adım 3a): optik akış + kare-farkı görselleştirme.
add_executable(dtrack_stabilize stabilize.cpp)
target_link_libraries(dtrack_stabilize PRIVATE
  dtrack_core
  dtrack_io
  dtrack_stabilization
  Threads::Threads)

# Gyro→kamera eksen/FOV kalibrasyonu (Adım 3b-ii): OF hedefine karşı AxisMap arar.
add_executable(dtrack_calibrate calibrate.cpp)
target_link_libraries(dtrack_calibrate PRIVATE
  dtrack_core
  dtrack_io
  dtrack_stabilization
  Threads::Threads)

# Ufuk konvansiyon kalibrasyonu (#14): görüntü ufkuna karşı 144 konvansiyon arar.
add_executable(dtrack_horizon_calib horizon_calib.cpp)
target_link_libraries(dtrack_horizon_calib PRIVATE
  dtrack_core
  dtrack_io
  Threads::Threads)

# Tespit demo (Adım 4): stabilize + hibrit (kare-farkı+MOG2) tespit + kutu çizimi.
add_executable(dtrack_detect detect.cpp)
target_link_libraries(dtrack_detect PRIVATE
  dtrack_core
  dtrack_io
  dtrack_stabilization
  dtrack_detection
  Threads::Threads)

# Takip demo (Adım 5/P4): tam zincir stabilize+tespit+takip; yalnız Confirmed izler.
add_executable(dtrack_track track.cpp)
target_link_libraries(dtrack_track PRIVATE
  dtrack_core
  dtrack_io
  dtrack_stabilization
  dtrack_detection
  dtrack_tracking
  Threads::Threads)
