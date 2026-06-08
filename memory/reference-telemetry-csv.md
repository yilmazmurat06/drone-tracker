---
name: reference-telemetry-csv
description: Liftoff telemetri CSV şeması ve video-telemetri senkron yöntemi (data/ uçuşları)
metadata: 
  node_type: memory
  type: reference
  originSessionId: cf2c3044-9c99-49c9-9ffc-fea607ff4d12
---

`data/flight_XX.telemetry.csv` — Liftoff FPV simülatöründen kayıtlı telemetri.

- **28 sütun, ~100 Hz**, 3 dosyada aynı şema. Header:
  `t_rel, epoch, packet_bytes, sim_time, pos_x, pos_y, pos_z, att_x, att_y, att_z, att_w,
   vel_x, vel_y, vel_z, gyro_pitch, gyro_roll, gyro_yaw, in_throttle, in_yaw, in_pitch,
   in_roll, batt_voltage, batt_pct, armed, motor_0, motor_1, motor_2, motor_3`
- **Zaman:** `t_rel` = video-hizalı saniye (`t_rel=0 ≈ video başı`, negatif=pre-roll);
  `epoch`=duvar saati; `sim_time`=Liftoff iç saati.
- **gyro_*** = açısal hız **DERECE/saniye** (rad/s DEĞİL; quaternion çapraz-doğrulamasıyla
  saptandı, oran tam π/180 çıktı — dtrack_calibrate). P1 stabilizasyonu rad'a çevirip kullanır.
  Tipik |gyro|~1–22, maks ~110 (deg/s makul; rad/s olsa imkânsızdı). **att_*** = attitude
  quaternion (driftsiz mutlak yönelim → P1 doğrulama/kısayol). **pos_*/vel_*** = dünya çerçevesi.

**Video:** her uçuş `flight_XX.mp4`, 1920×1080, **60 fps** (01: 104s/6265 kare, 02: 58s/3498,
03: 50s/2986). mp4'ler büyük (~yüzlerce MB) → `.gitignore`'da, git'e girmez.

**Senkron:** kare `i` için `t = i/60` (saniye) → telemetriyi bu `t_rel`'e interpolasyonla
eşle (gyro lineer, quaternion SLERP). Telemetri (100Hz) videodan (60fps) daha yoğun.

**Gelecek (gerçek Liftoff):** UDP telemetri `TelemetryConfiguration.Json` ile açılır
(GyroPitch/Roll/Yaw vb., örn. 127.0.0.1:9001); yalnız kendi uçurduğun drone için gelir.

İlgili: [[project-drone-tracker-decisions]]
