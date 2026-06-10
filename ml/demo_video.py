#!/usr/bin/env python3
"""Egitilmis 128^2 YOLO'yu tam video uzerinde goster.
Her GT kutusu etrafinda 128^2 kesit; tahmin + GT birlikte HUD'da.
Klavye: SPC=duraklat, q=cik, s=ekran goruntusu kaydet.

Kullanim:
  python3 ml/demo_video.py [--flight 01] [--conf 0.25] [--start 2000]
"""
import argparse, csv
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CROP = 128

def iou_val(a, b):
    ax1,ay1 = a[0]+a[2], a[1]+a[3]; bx1,by1 = b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax1,bx1)-max(a[0],b[0])); ih = max(0, min(ay1,by1)-max(a[1],b[1]))
    inter = iw*ih; u = a[2]*a[3]+b[2]*b[3]-inter
    return inter/u if u>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", default="01")
    ap.add_argument("--model", default=str(ROOT/"ml/onnx/yolo11_drone_trained.onnx"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--save", default="", help="video kaydet (mp4)")
    a = ap.parse_args()

    flight = f"flight_{a.flight}_{'084727' if a.flight=='01' else '084915' if a.flight=='02' else '085014'}"
    csv_path = ROOT / f"data/pseudo/{flight}_clean.csv"
    vid_path = ROOT / f"data/{flight}.mp4"
    model = YOLO(a.model, task="detect")

    gt_map = defaultdict(list)
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path)):
            gt_map[int(r["frame_idx"])].append(
                (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])))

    cap = cv2.VideoCapture(str(vid_path))
    if a.start: cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if a.save:
        writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (W, H))

    cv2.namedWindow("YOLO Liftoff", cv2.WINDOW_NORMAL)
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok: break
        fi = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - (0 if paused else 0)
        vis = frame.copy()

        gts = gt_map.get(fi, [])
        for (gx,gy,gw,gh) in gts:
            # kesit al
            side = CROP if max(gw,gh) <= CROP*0.75 else int(max(gw,gh)*1.6)
            side = min(side, W, H)
            x0 = int(np.clip(gx+gw/2-side/2, 0, W-side))
            y0 = int(np.clip(gy+gh/2-side/2, 0, H-side))
            patch = frame[y0:y0+side, x0:x0+side]
            s = CROP/side
            if side != CROP:
                patch = cv2.resize(patch,(CROP,CROP),interpolation=cv2.INTER_AREA)

            # GT kutusu (yesil)
            cv2.rectangle(vis,(gx,gy),(gx+gw,gy+gh),(0,220,0),2)
            cv2.putText(vis,"GT",( gx,gy-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,220,0),1)

            # kesit siniri (soluk mavi)
            cv2.rectangle(vis,(x0,y0),(x0+side,y0+side),(180,120,0),1)

            # YOLO tahmini
            res = model.predict(patch, imgsz=CROP, conf=a.conf, verbose=False)[0]
            for b in res.boxes:
                bx0,by0,bx1,by1 = (v/s for v in b.xyxy[0].tolist())
                px0=int(x0+bx0); py0=int(y0+by0); px1=int(x0+bx1); py1=int(y0+by1)
                conf = float(b.conf[0])
                pred = (int(x0+bx0/s), int(y0+by0/s),
                        int((bx1-bx0)/s), int((by1-by0)/s))
                gt_b = (gx,gy,gw,gh)
                iou_v = iou_val(gt_b, pred)
                col = (0,255,255) if iou_v>=0.3 else (0,80,255)
                cv2.rectangle(vis,(px0,py0),(px1,py1),col,2)
                cv2.putText(vis,f"{conf:.2f} IoU{iou_v:.2f}",(px0,py0-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)

        cv2.putText(vis, f"kare:{fi}  GT:{len(gts)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(vis, "SPC=duraklat  q=cik  s=kaydet",
                    (10,H-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        cv2.imshow("YOLO Liftoff", vis)
        if writer: writer.write(vis)

        k = cv2.waitKey(1 if not paused else 50) & 0xFF
        if   k == ord('q') or k == 27: break
        elif k == ord(' '): paused = not paused
        elif k == ord('s'):
            fn = f"ml/screenshot_{fi}.jpg"
            cv2.imwrite(fn, vis); print("kaydedildi:", fn)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
