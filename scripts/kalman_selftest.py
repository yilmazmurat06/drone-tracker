# -*- coding: utf-8 -*-
"""
kalman_selftest.py — screen_track.py'deki Kalman'i (Kf2 + KalmanTracker) DOGRULAR.

C++ test_kalman_math.cpp'nin Python karsiligi + ek bir blob-takip senaryosu.
Pencere/ekran/kumanda gerektirmez; saf sayisal. Kosma:
    python scripts/kalman_selftest.py

Sinananlar:
  A) Kararlilik: Kf2 P'si her adimda pozitif-tanimli kalir (NaN/patlama yok).
  B) Kararli-durum kazanci k0 ≈ α (≈0.55) -> Kalman ≈ kanitlanmis α-β.
  C) Dogruluk: gurultulu CV+manevra yorungesinde konum RMS dusuk.
  D) FPS-degismezlik: 30/60/120 fps'te raporlanan hiz ~ayni.
  E) KalmanTracker: hareketli hedefe MOG2-benzeri bloblarla kilit surekliligi
     (arada tespit bosluklari olsa bile coast edip kilidi korur).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from screen_track import Kf2, KalmanTracker  # noqa: E402

MEAS_STD  = 1.5
ACCEL_STD = 1500.0
R = MEAS_STD * MEAS_STD
Q = ACCEL_STD * ACCEL_STD

_fail = 0
def check(cond, msg):
    global _fail
    if not cond:
        print(f"  FAIL: {msg}")
        _fail += 1

def spd_ok(k):
    det = k.P00 * k.P11 - k.P01 * k.P10
    return (math.isfinite(k.P00) and math.isfinite(k.P11) and
            k.P00 > 0.0 and k.P11 > 0.0 and det > 0.0)


def test_core():
    import random
    rng = random.Random(12345)
    fps = 60.0; dt = 1.0 / fps
    k = Kf2()
    p0 = 100.0
    k.init(p0 + rng.gauss(0, MEAS_STD), 0.0, R, 60.0 * 60.0)
    t = 0.0; sum_e2 = 0.0; n = 0; spd_fail = 0; last_k0 = 0.0
    for i in range(600):
        t += dt
        truth = p0 + 80.0 * t + 40.0 * math.sin(2.0 * t)
        k.predict(dt, Q)
        if not spd_ok(k): spd_fail += 1
        last_k0 = k.P00 / (k.P00 + R)
        k.correct(truth + rng.gauss(0, MEAS_STD), R)
        if not spd_ok(k): spd_fail += 1
        if i > 150:
            e = k.p - truth; sum_e2 += e * e; n += 1
    rms = math.sqrt(sum_e2 / n)
    print(f"   [A] kararlilik: P-SPD ihlali={spd_fail} (ideal 0)")
    print(f"   [B] kararli-durum kazanci k0={last_k0:.3f} (α-β'de α≈0.55)")
    print(f"   [C] konum RMS={rms:.3f} px (σ_r={MEAS_STD})")
    check(spd_fail == 0, "P pozitif-tanimli kalmali")
    check(0.40 < last_k0 < 0.70, "kazanc α≈0.55 bandinda olmali")
    check(rms < 2.0 * MEAS_STD, "RMS olcum gurultusu mertebesinde olmali")


def test_fps_invariance():
    import random
    def run(fps):
        rng = random.Random(777)
        dt = 1.0 / fps
        k = Kf2(); k.init(0.0, 0.0, R, 60.0 * 60.0)
        t = 0.0; sv = 0.0; n = 0
        for _ in range(int(3.0 * fps)):
            t += dt
            truth = 90.0 * t
            k.predict(dt, Q)
            k.correct(truth + rng.gauss(0, MEAS_STD), R)
            if t > 1.0: sv += k.v; n += 1
        return sv / n if n else 0.0
    v30, v60, v120 = run(30.0), run(60.0), run(120.0)
    spread = max(v30, v60, v120) / max(1e-9, min(v30, v60, v120))
    print(f"   [D] hiz: 30={v30:.1f} 60={v60:.1f} 120={v120:.1f} px/s  yayilim={spread:.3f} (gercek=90)")
    check(70 < v30 < 110 and 70 < v60 < 110 and 70 < v120 < 110, "hiz ~90 px/s olmali")
    check(spread < 1.20, "fps'ten ~bagimsiz olmali")


def test_tracker_continuity():
    """Hareketli hedef + MOG2-benzeri bloblar; arada tespit boslugu. Kilit surekliligi."""
    import random
    rng = random.Random(2024)
    fps = 60.0; dt = 1.0 / fps
    trk = KalmanTracker()
    # Hedef: capraz hareket + hafif manevra.
    def truth_at(i):
        t = i * dt
        return (200.0 + 120.0 * t + 25.0 * math.sin(3.0 * t),
                150.0 + 90.0 * t)
    tx, ty = truth_at(0)
    trk.init(tx, ty, 12, 12)
    eval_n = 0; locked_n = 0; sum_e = 0.0
    gap = set(range(90, 96))  # 6 kare tespit boslugu
    for i in range(1, 240):
        gx, gy = truth_at(i)
        blobs = []
        if i not in gap:
            # Hedef blogu (gurultulu centroid) + birkaç celduktan (distractor).
            cx = gx + rng.gauss(0, MEAS_STD); cy = gy + rng.gauss(0, MEAS_STD)
            blobs.append((int(cx - 6), int(cy - 6), 12, 12, cx, cy, 144))
            for _ in range(2):  # alakasiz hareketli blob (clutter)
                dx = rng.uniform(-300, 300); dy = rng.uniform(-200, 200)
                if abs(dx) < 60 and abs(dy) < 60: dx += 120  # hedefe cok yakin olmasin
                bx, by = gx + dx, gy + dy
                blobs.append((int(bx - 5), int(by - 5), 10, 10, bx, by, 100))
        ok, box = trk.update(blobs, dt)
        c = trk.center
        e = math.hypot(c[0] - gx, c[1] - gy)
        eval_n += 1
        if ok and e <= 8.0:
            locked_n += 1; sum_e += e
    cont = locked_n / eval_n
    mean_e = sum_e / locked_n if locked_n else -1
    print(f"   [E] KalmanTracker: kilit surekliligi={cont:.2f}  konum hatasi={mean_e:.2f} px"
          f"  ({locked_n}/{eval_n})")
    check(cont > 0.90, "kilit surekliligi > 0.90 olmali (bosluk dahil)")
    check(0 <= mean_e < 4.0, "konum hatasi kucuk olmali")


if __name__ == "__main__":
    print("kalman_selftest (screen_track.py Kf2 + KalmanTracker)")
    test_core()
    test_fps_invariance()
    test_tracker_continuity()
    if _fail == 0:
        print("TUM TESTLER GECTI")
        sys.exit(0)
    print(f"{_fail} TEST BASARISIZ")
    sys.exit(1)
