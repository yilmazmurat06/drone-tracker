# -*- coding: utf-8 -*-
"""
screen_track.py — EKRAN-YAKALAMA tabanli CANLI cued takip (Liftoff vb. icin).

TAKIM:
  MOG2 (hareket tespiti) + KALMAN (CV) tahmini  <-- C++ pipeline ile ayni mantik
  CSRT KULLANILMIYOR: FPV kamerasi dondugunde arka plani hedefe kariştiriyor.
  Bunun yerine: her karede MOG2 hareketli blob listesi alinir; gercek 4-durumlu
  sabit-hiz Kalman filtresi (her eksen bir Kf2, C++ kalman_core.hpp ile birebir)
  NIS/Mahalanobis kapisi icinde en yakin blogu secip Joseph-form kovaryansla
  duzeltir ve tahmin yurutur. Hedef birkaç kare kaybolursa "coasting" yapar
  (P buyur -> kapi DOGAL genisler), fazla kaybolursa kilit duser ve en yakin
  harekete yeniden kilit atar.  (--filter ab ile alfa-beta yedegine gecilir.)

KILIT:
  - FARE ile surukle->birak : secilen bolgede kilit
  - f / ENTER veya SWITCH ASAGI : ekran merkezindeki en yakin harekete kilitle
TUSLAR:
  r=sifirla | SPACE=dur | s=kaydet | q/ESC=cik | +/-=blob esigi ayarla
  SWITCH YUKARI -> kilidi sifirla

Kullanim:
  python scripts/screen_track.py                 # birincil monitor
  python scripts/screen_track.py --monitor 2     # 2. monitor (Liftoff)
  python scripts/screen_track.py --width 1280    # gosterim genisligi
  python scripts/screen_track.py --save out.mp4  # oturum kaydi
"""

import argparse
import ctypes
from ctypes import wintypes
import time
import numpy as np
import cv2
import mss


# ================================================================
#  KUMANDA (JOYSTICK) — Windows Multimedia API
# ================================================================
class JOYINFOEX(ctypes.Structure):
    _fields_ = [
        ("dwSize",         wintypes.DWORD), ("dwFlags",        wintypes.DWORD),
        ("dwXpos",         wintypes.DWORD), ("dwYpos",         wintypes.DWORD),
        ("dwZpos",         wintypes.DWORD), ("dwRpos",         wintypes.DWORD),
        ("dwUpos",         wintypes.DWORD), ("dwVpos",         wintypes.DWORD),
        ("dwButtons",      wintypes.DWORD), ("dwButtonNumber", wintypes.DWORD),
        ("dwPOV",          wintypes.DWORD), ("dwReserved1",    wintypes.DWORD),
        ("dwReserved2",    wintypes.DWORD),
    ]

_winmm = ctypes.windll.winmm
SWITCH_AXIS     = "V"
SWITCH_LOW_THR  = 10000
SWITCH_HIGH_THR = 50000

def _read_joy(joy_id):
    info = JOYINFOEX()
    info.dwSize  = ctypes.sizeof(JOYINFOEX)
    info.dwFlags = 0x00FF
    if _winmm.joyGetPosEx(joy_id, ctypes.byref(info)) != 0:
        return None
    return {"X": info.dwXpos, "Y": info.dwYpos, "Z": info.dwZpos,
            "R": info.dwRpos, "U": info.dwUpos, "V": info.dwVpos,
            "btn": info.dwButtons}

class JoySwitch:
    def __init__(self):
        self.joy_id = -1
        for i in range(8):
            if _read_joy(i) is not None:
                self.joy_id = i; break
        self._prev_low = False
        if self.joy_id >= 0:
            s = _read_joy(self.joy_id)
            if s: self._prev_low = s[SWITCH_AXIS] < SWITCH_LOW_THR
            print(f"  [kumanda] slot={self.joy_id}, eksen='{SWITCH_AXIS}'")
        else:
            print("  [kumanda] UYARI: joystick bulunamadi.")

    def poll(self):
        if self.joy_id < 0: return None
        s = _read_joy(self.joy_id)
        if s is None: return None
        v = s[SWITCH_AXIS]
        now_low  = v < SWITCH_LOW_THR
        now_high = v > SWITCH_HIGH_THR
        ev = None
        if now_low  and not self._prev_low: ev = "down"
        elif now_high and self._prev_low:   ev = "up"
        if now_low or now_high: self._prev_low = now_low
        return ev


# ================================================================
#  MOG2 HAREKET DEDEKTORU
#  C++ MogDetector'un Python karsiligi.
#  history, varThreshold ve min/max alan parametreleri runtime'da ayarlanabilir.
# ================================================================
class MotionDetector:
    def __init__(self, history=120, var_threshold=36, learning_rate=0.005):
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.lr = learning_rate
        self.min_area = 8
        self.max_area = 6000
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, gray):
        """gray: CV_8UC1  ->  (blobs, mask)
        blobs: [(x, y, w, h, cx, cy, area), ...]  sol-ust kose + boyut + merkez"""
        mask = self.mog.apply(gray, learningRate=self.lr)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._k3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._k7)
        n, _, st, ct = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if a < self.min_area or a > self.max_area: continue
            x = st[i, cv2.CC_STAT_LEFT];  y = st[i, cv2.CC_STAT_TOP]
            w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
            blobs.append((x, y, w, h, float(ct[i, 0]), float(ct[i, 1]), a))
        return blobs, mask


# ================================================================
#  KALMAN ÇEKİRDEĞİ (Kf2) — C++ kalman_core.hpp ile BİREBİR.
#  1B sabit-hız (CV) Kalman: durum [konum, hız]. Köşegen ölçüm gürültüsüyle
#  4-durumlu CV filtresi iki bağımsız 2-durumlu eksene ayrışır -> her iz iki Kf2.
#  Joseph-form kovaryans (sayısal kararlı) + beyaz-gürültü ivme süreç modeli.
# ================================================================
class Kf2:
    __slots__ = ("p", "v", "P00", "P01", "P10", "P11")

    def __init__(self):
        self.p = 0.0; self.v = 0.0
        self.P00 = 1.0; self.P01 = 0.0; self.P10 = 0.0; self.P11 = 1.0

    def init(self, pos, vel, pos_var, vel_var):
        self.p = pos; self.v = vel
        self.P00 = pos_var; self.P01 = 0.0; self.P10 = 0.0; self.P11 = vel_var

    def predict(self, dt, q):
        # x = F x ; P = F P Fᵀ + Q  (q = σ_a², beyaz-gürültü ivme).
        self.p += self.v * dt
        a = self.P00 + dt * (self.P01 + self.P10) + dt * dt * self.P11
        b = self.P01 + dt * self.P11
        c = self.P10 + dt * self.P11
        d = self.P11
        dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt2 * dt2
        self.P00 = a + q * dt4 * 0.25
        self.P01 = b + q * dt3 * 0.5
        self.P10 = c + q * dt3 * 0.5
        self.P11 = d + q * dt2

    def nis(self, z, r):
        # NIS = y²/S (1-dof, eksen başına). S = P00 + r.
        y = z - self.p
        return (y * y) / (self.P00 + r)

    def correct(self, z, r):
        # Joseph-form güncelleme (P daima simetrik+pozitif). K = P Hᵀ S⁻¹, H=[1,0].
        s = self.P00 + r
        k0 = self.P00 / s; k1 = self.P10 / s
        y = z - self.p
        self.p += k0 * y; self.v += k1 * y
        ic = 1.0 - k0
        m00 = ic * self.P00; m01 = ic * self.P01
        m10 = self.P10 - k1 * self.P00; m11 = self.P11 - k1 * self.P01
        self.P00 = m00 * ic + r * k0 * k0
        self.P01 = -k1 * m00 + m01 + r * k0 * k1
        self.P10 = m10 * ic + r * k0 * k1
        self.P11 = -k1 * m10 + m11 + r * k1 * k1


# ================================================================
#  KALMAN TAKİPÇİ (VARSAYILAN) — C++ KalmanTracker ile aynı mantık.
#  Her karede MOG2 bloblarından NIS (Mahalanobis²) kapısı içinde en yakınına
#  kilitlenir; yoksa coast eder. Coast'ta P büyür -> kapı DOĞAL genişler
#  (elle eşik ayarı gerekmez). Hız px/SANİYE biriminde (fps'ten bağımsız).
# ================================================================
class KalmanTracker:
    def __init__(self, meas_std=1.5, accel_std=1500.0, init_vel_std=90.0,
                 gate_nis=13.8, coast_max=15):
        self.r = meas_std * meas_std            # σ_r² ölçüm varyansı
        self.q = accel_std * accel_std          # σ_a² süreç (manevra) gürültüsü
        self.init_vel_var = init_vel_std * init_vel_std
        self.gate_nis = gate_nis                # 2-dof χ²: 13.8 (%99.9)
        self.coast_max = coast_max
        self.kx = Kf2(); self.ky = Kf2()
        self.box_hw = np.array([20.0, 20.0])
        self.coasting = 0
        self.active = False

    def init(self, cx, cy, w, h):
        # Tek-nokta kilit: hız bilinmiyor (0), P0 ÖLÇÜLÜ (salınım yok, §2.7).
        self.kx.init(cx, 0.0, self.r, self.init_vel_var)
        self.ky.init(cy, 0.0, self.r, self.init_vel_var)
        self.box_hw = np.array([max(8.0, w) / 2.0, max(8.0, h) / 2.0])
        self.coasting = 0
        self.active = True

    def update(self, blobs, dt):
        """MOG2 bloblarindan NIS kapisi icinde en yakinina kilitlen; (kilitli, kutu)."""
        if not self.active:
            return False, (0, 0, 1, 1)
        self.kx.predict(dt, self.q); self.ky.predict(dt, self.q)
        best = None; best_nis = self.gate_nis
        for b in blobs:
            nis = self.kx.nis(b[4], self.r) + self.ky.nis(b[5], self.r)
            if nis < best_nis:
                best_nis = nis; best = b
        if best is not None:
            self.kx.correct(best[4], self.r); self.ky.correct(best[5], self.r)
            self.box_hw = 0.85 * self.box_hw + 0.15 * np.array([best[2] / 2.0, best[3] / 2.0])
            self.coasting = 0
        else:
            self.coasting += 1   # coast: sadece predict (P buyur -> kapi genisler)
        return (self.coasting <= self.coast_max), self._box()

    def _box(self):
        w = max(8, int(self.box_hw[0] * 2)); h = max(8, int(self.box_hw[1] * 2))
        return (int(self.kx.p - self.box_hw[0]), int(self.ky.p - self.box_hw[1]), w, h)

    @property
    def center(self):
        return np.array([self.kx.p, self.ky.p])

    @property
    def vel(self):           # px/saniye
        return np.array([self.kx.v, self.ky.v])

    @property
    def is_coasting(self):
        return self.coasting > 0


# ================================================================
#  ALPHA-BETA TAKİPÇİ (YEDEK; --filter ab) — sabit-kazanç alternatifi.
#  Kararlı-durumda Kalman'a denktir; tuning gerektirmez. Hız px/KARE.
# ================================================================
class AlphaBetaTracker:
    def __init__(self, alpha=0.55, beta=0.18, gate_px=90, coast_max=12):
        self.alpha     = alpha
        self.beta      = beta
        self.gate      = gate_px
        self.coast_max = coast_max
        self.pos       = None      # np.array([cx, cy])
        self._vel      = np.zeros(2)
        self.box_hw    = np.array([20.0, 20.0])
        self.coasting  = 0
        self.active    = False

    def init(self, cx, cy, w, h):
        self.pos     = np.array([cx, cy], dtype=float)
        self._vel    = np.zeros(2)
        self.box_hw  = np.array([max(8.0, w) / 2.0, max(8.0, h) / 2.0])
        self.coasting = 0
        self.active   = True

    def update(self, blobs, dt=None):  # dt yok sayılır (hız px/kare)
        if self.pos is None:
            return False, (0, 0, 1, 1)
        pred = self.pos + self._vel
        gate = self.gate * (1.0 + 0.4 * self.coasting)  # coast'ta elle genişlet
        best = None; bd = gate
        for b in blobs:
            d = np.hypot(b[4] - pred[0], b[5] - pred[1])
            if d < bd: bd = d; best = b
        if best is not None:
            meas  = np.array([best[4], best[5]])
            innov = meas - pred
            self._vel = self._vel + self.beta  * innov
            self.pos  = pred      + self.alpha * innov
            self.box_hw = 0.85 * self.box_hw + 0.15 * np.array([best[2] / 2.0, best[3] / 2.0])
            self.coasting = 0
        else:
            self.pos = pred
            self.coasting += 1
        return (self.coasting <= self.coast_max), self._box()

    def _box(self):
        if self.pos is None: return (0, 0, 1, 1)
        w = max(8, int(self.box_hw[0] * 2))
        h = max(8, int(self.box_hw[1] * 2))
        return (int(self.pos[0] - self.box_hw[0]), int(self.pos[1] - self.box_hw[1]), w, h)

    @property
    def center(self):
        return self.pos.copy() if self.pos is not None else np.zeros(2)

    @property
    def vel(self):           # px/kare
        return self._vel

    @property
    def is_coasting(self):
        return self.coasting > 0


# ================================================================
#  YARDIMCI FONKSİYONLAR
# ================================================================
class Mouse:
    def __init__(self):
        self.dragging = self.ready = False
        self.p0 = self.p1 = (0, 0)

def rect_from_points(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            abs(a[0] - b[0]), abs(a[1] - b[1]))

def draw_crosshair(vis):
    h, w = vis.shape[:2]
    c = (w // 2, h // 2); R = max(28, h // 10); g = R // 3
    col = (0, 255, 0)
    cv2.circle(vis, c, R, col, 1, cv2.LINE_AA)
    cv2.circle(vis, c, 2, col, -1, cv2.LINE_AA)
    cv2.line(vis, (c[0] - R - 12, c[1]), (c[0] - g, c[1]), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0] + g, c[1]), (c[0] + R + 12, c[1]), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0], c[1] - R - 12), (c[0], c[1] - g), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0], c[1] + g), (c[0], c[1] + R + 12), col, 1, cv2.LINE_AA)

def nearest_blob(blobs, cx, cy, max_dist=1e9):
    best = None; bd = max_dist
    for b in blobs:
        d = np.hypot(b[4] - cx, b[5] - cy)
        if d < bd: bd = d; best = b
    return best

def pick_window_pos(monitors, cap_idx):
    for i in range(1, len(monitors)):
        if i == cap_idx: continue
        m = monitors[i]; return (m["left"] + 40, m["top"] + 40)
    m = monitors[cap_idx]; return (m["left"] + 40, m["top"] + 40)


# ================================================================
#  ANA DONGU
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--region",  type=str, default="")
    ap.add_argument("--width",   type=int, default=1280)
    ap.add_argument("--fps",     type=float, default=60.0)
    ap.add_argument("--save",    type=str, default="")
    ap.add_argument("--win-pos", type=str, default="")
    ap.add_argument("--filter",  type=str, default="kalman",
                    choices=["kalman", "ab"], help="kalman (varsayilan) | ab (alfa-beta yedek)")
    args = ap.parse_args()

    sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
    monitors = sct.monitors
    print("Monitorler:")
    for i in range(1, len(monitors)):
        m = monitors[i]
        print(f"  [{i}] {m['width']}x{m['height']} @ ({m['left']},{m['top']})")

    if args.region:
        x, y, w, h = [int(v) for v in args.region.split(",")]
        grab = {"left": x, "top": y, "width": w, "height": h}; cap_idx = -1
    else:
        cap_idx = args.monitor if 0 < args.monitor < len(monitors) else 1
        grab = monitors[cap_idx]

    WIN = "screen_track | MOG2+Kalman | f/ENTER/SW=kilit | r=sifirla | +/-=hassasiyet | q=cik"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    mouse = Mouse()
    def on_mouse(ev, x, y, fl, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse.dragging = True; mouse.ready = False; mouse.p0 = mouse.p1 = (x, y)
        elif ev == cv2.EVENT_MOUSEMOVE and mouse.dragging:
            mouse.p1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            mouse.dragging = False; mouse.p1 = (x, y); mouse.ready = True
    cv2.setMouseCallback(WIN, on_mouse)

    if args.win_pos:
        wx, wy = [int(v) for v in args.win_pos.split(",")]
    else:
        wx, wy = pick_window_pos(monitors, cap_idx if cap_idx > 0 else 1)
    try: cv2.moveWindow(WIN, wx, wy)
    except Exception: pass

    joy     = JoySwitch()
    mog     = MotionDetector()
    use_kalman = (args.filter == "kalman")
    tracker = KalmanTracker() if use_kalman else AlphaBetaTracker()
    flt_name = "Kalman" if use_kalman else "AlphaBeta"
    arrow_scale = 0.15 if use_kalman else 5.0   # px/s -> kisa ok ; px/kare -> uzun ok
    locked  = False      # aktif kilit var mi
    box     = None       # son kare kutusu (x,y,w,h)
    paused  = False
    writer  = None; snap = 0
    ms_hist = []
    min_interval = 1.0 / max(1.0, args.fps)

    print(f"=== screen_track (MOG2 + {flt_name}) ===")
    print(f"  yakalanan: {grab['width']}x{grab['height']} @ ({grab['left']},{grab['top']})")
    print("  f/ENTER/SWITCH=kilit | r=sifirla | +/-=blob esigi | SPACE=dur | q=cik")
    print("  Sari=MOG2 blob (hareketli nesne) | Yesil=kilitli | Turuncu=coasting")

    last_t = 0.0
    dt_frame = min_interval   # filtre icin gercek kare-arasi sure (saniye)
    proc   = None
    blobs  = []

    while True:
        if not paused:
            now = time.time()
            sleep_dt = min_interval - (now - last_t)
            if sleep_dt > 0: time.sleep(sleep_dt)
            t_now = time.time()
            # Filtre icin GERCEK kare-arasi sure (saniye); [1ms, 100ms] araliginda kirp.
            dt_frame = min(0.1, max(1e-3, t_now - last_t)) if last_t > 0 else min_interval
            last_t = t_now

            raw   = np.asarray(sct.grab(grab))
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            scale = args.width / frame.shape[1]
            proc  = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA) if scale != 1.0 else frame

        if proc is None: continue
        ph, pw = proc.shape[:2]
        gray   = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

        # MOG2 her zaman çalışır
        blobs, motion_mask = mog.update(gray)

        # --- Kumanda switch ---
        sw = joy.poll()
        if sw == "down":
            cx, cy = pw / 2, ph / 2
            b = nearest_blob(blobs, cx, cy, max_dist=min(pw, ph) / 2)
            if b:
                tracker.init(b[4], b[5], b[2], b[3])
            else:
                d = max(40, pw // 20)
                tracker.init(cx, cy, d, d)
            locked = True
            print(f"  [SWITCH-KILIT] konum={tracker.center}")
        elif sw == "up":
            locked = False; box = None
            print("  [SWITCH-SIFIRLA]")

        # --- Fare secimi ---
        if mouse.ready:
            r = rect_from_points(mouse.p0, mouse.p1); mouse.ready = False
            if r[2] > 6 and r[3] > 6:
                rcx = r[0] + r[2] / 2.0; rcy = r[1] + r[3] / 2.0
                # Secili kutunun icindeki en yakin blogu bul; yoksa kutu merkezini kullan
                in_roi = [b for b in blobs
                          if r[0] <= b[4] <= r[0]+r[2] and r[1] <= b[5] <= r[1]+r[3]]
                if in_roi:
                    b = min(in_roi, key=lambda b: np.hypot(b[4]-rcx, b[5]-rcy))
                    tracker.init(b[4], b[5], b[2], b[3])
                else:
                    tracker.init(rcx, rcy, r[2], r[3])
                locked = True
                print(f"  [FARE-KILIT] konum={tracker.center}")

        # --- AlphaBeta guncelle ---
        ms = 0.0
        if locked and not paused:
            t0 = time.time()
            ok, bb = tracker.update(blobs, dt_frame)
            ms = (time.time() - t0) * 1000.0
            ms_hist.append(ms)
            if len(ms_hist) > 30: ms_hist.pop(0)

            if ok:
                box = bb
            else:
                # Coast_max asildi: en yakin harekete otomatik yeniden kilit
                c = tracker.center
                b = nearest_blob(blobs, c[0], c[1], max_dist=200)
                if b:
                    tracker.init(b[4], b[5], b[2], b[3])
                    locked = True
                    print(f"  [YENIDEN-KILIT] blob: ({b[4]:.0f},{b[5]:.0f})")
                else:
                    locked = False; box = None
                    print("  [KAYIP] hareketli blob bulunamadi")

        # --- Gorselleştirme ---
        vis = proc.copy()

        # MOG2 hareket katmani (yari saydam turuncu)
        if np.any(motion_mask):
            overlay = np.zeros_like(vis)
            overlay[motion_mask > 0] = (0, 180, 255)
            cv2.addWeighted(vis, 1.0, overlay, 0.30, 0, vis)

        # Her MOG2 blobu icin ince sari cerceve
        for b in blobs:
            cv2.rectangle(vis, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]),
                          (0, 200, 255), 1, cv2.LINE_AA)

        # Fare suruklemesi (beyaz kesik cizgi)
        if mouse.dragging:
            r = rect_from_points(mouse.p0, mouse.p1)
            cv2.rectangle(vis, (r[0], r[1]), (r[0]+r[2], r[1]+r[3]), (255, 255, 255), 1)

        # Takip kutusu
        if locked and box is not None:
            coasting = tracker.is_coasting
            col = (0, 140, 255) if coasting else (0, 255, 0)   # turuncu=coasting, yesil=kilitli
            cv2.rectangle(vis, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), col, 2)
            cc = (int(tracker.center[0]), int(tracker.center[1]))
            cv2.drawMarker(vis, cc, col, cv2.MARKER_CROSS, 16, 1)
            spd = float(np.hypot(tracker.vel[0], tracker.vel[1]))
            if spd > 0.5:
                tip = (cc[0] + int(tracker.vel[0] * arrow_scale),
                       cc[1] + int(tracker.vel[1] * arrow_scale))
                cv2.arrowedLine(vis, cc, tip, (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.3)
            status = f"COAST({tracker.coasting})" if coasting else "KILITLI"
            cv2.putText(vis, status, (box[0], max(0, box[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

        # HUD
        avg_ms = sum(ms_hist) / len(ms_hist) if ms_hist else 0.0
        fps_s  = f"~{1000/avg_ms:.0f}FPS" if avg_ms > 1 else ""
        l1 = (f"MOG2+{flt_name} | {avg_ms:.1f}ms {fps_s}"
              f"  blob:{len(blobs)}"
              f"  esik:{mog.mog.getVarThreshold():.0f}"
              f"{'  [DURDU]' if paused else ''}")
        l2 = "f/ENTER/SW=kilit | r=sifirla | +/-=blob esigi | SPACE=dur | q=cik"
        cv2.rectangle(vis, (0, 0), (pw, 52), (0, 0, 0), -1)
        cv2.putText(vis, l1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, l2, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 220, 180), 1, cv2.LINE_AA)
        draw_crosshair(vis)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         30.0, (vis.shape[1], vis.shape[0]), True)
            writer.write(vis)

        cv2.imshow(WIN, vis)
        key = cv2.waitKey(1) & 0xFF
        if   key in (ord('q'), 27): break
        elif key == ord(' '): paused = not paused
        elif key == ord('r'): locked = False; box = None; ms_hist.clear()
        elif key in (ord('f'), 13):
            cx, cy = pw / 2, ph / 2
            b = nearest_blob(blobs, cx, cy, max_dist=min(pw, ph) / 2)
            if b:
                tracker.init(b[4], b[5], b[2], b[3])
            else:
                d = max(40, pw // 20)
                tracker.init(cx, cy, d, d)
            locked = True
            print(f"  [MERKEZ-KILIT] konum={tracker.center}")
        elif key == ord('+') or key == ord('='):
            t = max(5, mog.mog.getVarThreshold() - 4)
            mog.mog.setVarThreshold(t); print(f"  [esik] {t:.0f}")
        elif key == ord('-'):
            t = min(200, mog.mog.getVarThreshold() + 4)
            mog.mog.setVarThreshold(t); print(f"  [esik] {t:.0f}")
        elif key == ord('s'):
            p = f"snap_{snap:03d}.png"; snap += 1
            cv2.imwrite(p, vis); print(f"  [kayit] {p}")

    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
