# -*- coding: utf-8 -*-
"""
screen_track.py — EKRAN-YAKALAMA tabanli CANLI cued takip (Liftoff vb. icin).

Fikir: Liftoff'u oynarken ekranda gordugun goruntuyu bir "ekran izleme araci"
(mss) ile yakalariz; CSRT korelasyon takipcisi hedefi canli izler ve uzerine
kutu + nisangah cizer. Bu, projedeki live_track.cpp'nin ("operator hedefi
isaretler, sistem kilitlenir") Python karsiligidir — derleyici/WSL/stream YOK.

KILIT:
  - FARE ile surukleyerek hedef kutusu ciz -> birak -> kilitlenir.
  - 'f' veya ENTER  -> ekran MERKEZINDEKI (nisangah) hedefe kilitlen.
TUSLAR:
  SPACE dur/devam | r kilidi sifirla | s kare-kaydet | [ ] pencere kucult/buyut |
  q veya ESC cik.

Kullanim:
  python scripts/screen_track.py                 # birincil monitoru yakala
  python scripts/screen_track.py --monitor 2     # 2. monitoru yakala
  python scripts/screen_track.py --width 1280    # isleme/gosterim genisligi
  python scripts/screen_track.py --region 100,100,1280,720   # sanal-masaustu bolgesi
  python scripts/screen_track.py --save out.mp4  # oturumu kaydet

NOT: Yakaladigin monitor ile tracker penceresinin OLDUGU monitor FARKLI olsun
(yoksa pencere kendini yakalar -> sonsuz ayna). Script pencereyi otomatik
olarak yakalanmayan bir monitore tasimaya calisir.
"""

import argparse
import time
import numpy as np
import cv2
import mss


# ----------------------- CSRT yardimcisi -----------------------
def make_csrt():
    # OpenCV 4.13: ust-seviye TrackerCSRT_create; eski surumlerde legacy altinda.
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("CSRT yok: 'pip install opencv-contrib-python' gerekir.")


# ----------------------- Fare durumu -----------------------
class Mouse:
    def __init__(self):
        self.dragging = False
        self.ready = False
        self.p0 = (0, 0)
        self.p1 = (0, 0)


def rect_from_points(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    return (x0, y0, x1 - x0, y1 - y0)


# ----------------------- Nisangah cizimi -----------------------
def draw_crosshair(vis):
    h, w = vis.shape[:2]
    c = (w // 2, h // 2)
    R = max(28, h // 10)
    gap = R // 3
    col = (0, 255, 0)
    cv2.circle(vis, c, R, col, 1, cv2.LINE_AA)
    cv2.circle(vis, c, 2, col, -1, cv2.LINE_AA)
    cv2.line(vis, (c[0] - R - 12, c[1]), (c[0] - gap, c[1]), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0] + gap, c[1]), (c[0] + R + 12, c[1]), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0], c[1] - R - 12), (c[0], c[1] - gap), col, 1, cv2.LINE_AA)
    cv2.line(vis, (c[0], c[1] + gap), (c[0], c[1] + R + 12), col, 1, cv2.LINE_AA)


# ----------------------- Merkez (nisangah) kilidi -----------------------
# ROI icinde gokyuzu/arka-plana karsi baskin KOMPAKT nesneyi segmentle -> kutu.
# Bulamazsa varsayilan boyutta merkez kutusu doner (yine de kilitlenir).
def lock_box_at_center(frame, ctr, scale_ref):
    h, w = frame.shape[:2]
    s = max(64, h // 8)
    x0 = max(0, ctr[0] - s // 2)
    y0 = max(0, ctr[1] - s // 2)
    x1 = min(w, ctr[0] + s // 2)
    y1 = min(h, ctr[1] + s // 2)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        d = max(44, w // 18)
        return (ctr[0] - d // 2, ctr[1] - d // 2, d, d)

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    rh, rw = g.shape[:2]
    bw, bh = max(2, rw // 8), max(2, rh // 8)
    # Kenar seridi = arka plan referansi.
    border = np.zeros_like(g)
    border[:bh, :] = 255; border[-bh:, :] = 255
    border[:, :bw] = 255; border[:, -bw:] = 255
    sky = cv2.mean(g, border)[0]

    fdiff = cv2.absdiff(g, np.full_like(g, int(sky)))
    obj = (fdiff > 16).astype(np.uint8) * 255
    gx = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_16S, 1, 0))
    gy = cv2.convertScaleAbs(cv2.Sobel(g, cv2.CV_16S, 0, 1))
    grad = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    obj = cv2.bitwise_or(obj, (grad > 40).astype(np.uint8) * 255)
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k9)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, k3)

    n, lab, st, ct = cv2.connectedComponentsWithStats(obj, connectivity=8)
    c0 = (rw / 2.0, rh / 2.0)
    best, best_score = -1, -1.0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < 25:
            continue
        dist = np.hypot(ct[i, 0] - c0[0], ct[i, 1] - c0[1])
        score = a - dist * 6.0
        if score > best_score:
            best_score, best = score, i
    if best >= 0:
        bx = st[best, cv2.CC_STAT_LEFT]; by = st[best, cv2.CC_STAT_TOP]
        bbw = st[best, cv2.CC_STAT_WIDTH]; bbh = st[best, cv2.CC_STAT_HEIGHT]
        good = 6 < bbw < 0.85 * rw and 6 < bbh < 0.85 * rh
        if good:
            return (x0 + bx, y0 + by, bbw, bbh)
    d = max(44, w // 18)
    return (max(0, ctr[0] - d // 2), max(0, ctr[1] - d // 2), d, d)


def pick_window_pos(monitors, cap_idx):
    """Yakalanan monitor disinda bir monitorun sol-ust kosesini dondur."""
    for i in range(1, len(monitors)):
        if i == cap_idx:
            continue
        m = monitors[i]
        return (m["left"] + 40, m["top"] + 40)
    m = monitors[cap_idx]
    return (m["left"] + 40, m["top"] + 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", type=int, default=1,
                    help="yakalanacak monitor (1=birincil, 2=ikincil...)")
    ap.add_argument("--region", type=str, default="",
                    help="sanal-masaustu bolgesi x,y,w,h (verilirse --monitor yok sayilir)")
    ap.add_argument("--width", type=int, default=1280, help="isleme/gosterim genisligi (px)")
    ap.add_argument("--fps", type=float, default=60.0, help="hedef yakalama FPS ust siniri")
    ap.add_argument("--save", type=str, default="", help="oturumu mp4 olarak kaydet")
    ap.add_argument("--win-pos", type=str, default="", help="pencere konumu x,y (override)")
    args = ap.parse_args()

    sct = mss.MSS() if hasattr(mss, "MSS") else mss.mss()
    monitors = sct.monitors  # [0]=hepsi, [1]=birincil, ...
    print("Monitorler:")
    for i in range(1, len(monitors)):
        m = monitors[i]
        print(f"  [{i}] {m['width']}x{m['height']} @ ({m['left']},{m['top']})")

    if args.region:
        x, y, w, h = [int(v) for v in args.region.split(",")]
        grab = {"left": x, "top": y, "width": w, "height": h}
        cap_idx = -1
    else:
        cap_idx = args.monitor if 0 < args.monitor < len(monitors) else 1
        grab = monitors[cap_idx]

    win = "screen_track  (FARE: kutu ciz | f/ENTER: merkeze kilitle | q: cik)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mouse = Mouse()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.dragging = True; mouse.ready = False
            mouse.p0 = (x, y); mouse.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse.dragging:
            mouse.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse.dragging = False; mouse.p1 = (x, y); mouse.ready = True
    cv2.setMouseCallback(win, on_mouse)

    # Pencereyi yakalanmayan monitore tasi (kendini yakalama -> ayna onlenir).
    if args.win_pos:
        wx, wy = [int(v) for v in args.win_pos.split(",")]
    else:
        wx, wy = pick_window_pos(monitors, cap_idx if cap_idx > 0 else 1)
    try:
        cv2.moveWindow(win, wx, wy)
    except Exception:
        pass

    tracker = None
    box = None              # (x,y,w,h) gosterim uzayinda
    center = None
    vel = np.array([0.0, 0.0])
    lost = False
    paused = False
    writer = None
    snap = 0
    ms_hist = []
    min_interval = 1.0 / max(1.0, args.fps)

    print("=== screen_track (CSRT) ===")
    print(f"  yakalanan: {grab['width']}x{grab['height']} @ ({grab['left']},{grab['top']})")
    print("  FARE ile hedefi kutuyla isaretle, ya da 'f'/ENTER ile merkeze kilitle.")

    last_t = 0.0
    while True:
        if not paused:
            # FPS sinirlama (CPU rahatlat).
            now = time.time()
            if now - last_t < min_interval:
                time.sleep(max(0.0, min_interval - (now - last_t)))
            last_t = time.time()

            raw = np.asarray(sct.grab(grab))  # BGRA
            frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
            scale = args.width / frame.shape[1]
            if scale != 1.0 and scale > 0:
                proc = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            else:
                proc = frame
        # paused: son proc'u koru
        ph, pw = proc.shape[:2]

        # Yeni fare secimi -> kilitle.
        if mouse.ready:
            r = rect_from_points(mouse.p0, mouse.p1)
            mouse.ready = False
            if r[2] > 6 and r[3] > 6:
                tracker = make_csrt()
                tracker.init(proc, tuple(int(v) for v in r))
                box = r; center = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0])
                vel = np.array([0.0, 0.0]); lost = False
                print(f"  [cue] kilit: {r}")

        # Takip guncelle.
        ms = 0.0
        if tracker is not None and not paused and not lost:
            t0 = time.time()
            ok, bb = tracker.update(proc)
            ms = (time.time() - t0) * 1000.0
            ms_hist.append(ms)
            if len(ms_hist) > 30:
                ms_hist.pop(0)
            if ok:
                nb = tuple(int(v) for v in bb)
                new_c = np.array([nb[0] + nb[2] / 2.0, nb[1] + nb[3] / 2.0])
                if center is not None:
                    vel = 0.6 * vel + 0.4 * (new_c - center)
                box = nb; center = new_c; lost = False
            else:
                lost = True

        # Cizim.
        vis = proc.copy()
        if mouse.dragging:
            r = rect_from_points(mouse.p0, mouse.p1)
            cv2.rectangle(vis, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]),
                          (0, 255, 255), 2)
        if box is not None:
            col = (0, 0, 255) if lost else (0, 255, 0)
            cv2.rectangle(vis, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), col, 2)
            if center is not None:
                cc = (int(center[0]), int(center[1]))
                cv2.drawMarker(vis, cc, col, cv2.MARKER_CROSS, 18, 1)
                if not lost:
                    tip = (cc[0] + int(vel[0] * 6), cc[1] + int(vel[1] * 6))
                    cv2.arrowedLine(vis, cc, tip, (255, 200, 0), 2, cv2.LINE_AA, tipLength=0.3)
            cv2.putText(vis, "KAYIP - yeniden sec (r/fare)" if lost else "KILITLI",
                        (box[0], max(0, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                        cv2.LINE_AA)

        avg_ms = sum(ms_hist) / len(ms_hist) if ms_hist else 0.0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        spd = float(np.hypot(vel[0], vel[1]))
        l1 = "takip %.1f ms (~%.0f FPS)%s | kutu %s | hiz %.0f px/k" % (
            avg_ms, fps, " [DURDU]" if paused else "",
            "%dx%d" % (box[2], box[3]) if box else "-", spd)
        l2 = ("FARE: yeni kutu | r: sifirla | SPACE: dur | q: cik" if tracker
              else "FARE ile hedefi isaretle | f/ENTER: merkeze kilitle | q: cik")
        cv2.rectangle(vis, (0, 0), (pw, 52), (0, 0, 0), -1)
        cv2.putText(vis, l1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(vis, l2, (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 180), 1, cv2.LINE_AA)
        draw_crosshair(vis)

        if args.save:
            if writer is None:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                         30.0, (vis.shape[1], vis.shape[0]), True)
            writer.write(vis)

        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (ord('f'), 13):  # nisangah (merkez) kilidi
            cc = (pw // 2, ph // 2)
            r = lock_box_at_center(proc, cc, scale)
            if r[2] > 6 and r[3] > 6:
                tracker = make_csrt()
                tracker.init(proc, tuple(int(v) for v in r))
                box = r; center = np.array([r[0] + r[2] / 2.0, r[1] + r[3] / 2.0])
                vel = np.array([0.0, 0.0]); lost = False
                print(f"  [nisangah-kilit] {r}")
        elif key == ord('r'):
            tracker = None; box = None; center = None; lost = False; ms_hist.clear()
        elif key == ord('s'):
            p = "screen_snap_%03d.png" % snap; snap += 1
            cv2.imwrite(p, vis); print(f"  [kayit] {p}")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if args.save:
        print(f"  kayit: {args.save}")


if __name__ == "__main__":
    main()
