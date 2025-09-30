# -*- coding: utf-8 -*-
import cv2, joblib, numpy as np, mediapipe as mp
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

# 字型（可簡化為英文顯示）
def put_text(img, text, xy, color=(0,255,0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# 從 collect_and_label 的邏輯借用：眼對齊 + 計算特徵（為簡潔，這裡直接引用相同公式）
from collect_and_label import affine_align_by_eyes, L_OUT,L_IN,L_UP,L_DN,R_OUT,R_IN,R_UP,R_DN, \
                              M_LEFT,M_RIGHT,M_UP,M_DN, LBROW,RBROW, dist, minmax_norm, \
                              mouth_curve_M, brow_raise_B, EAR, MAR

def compute_feats_for_frame(aligned, mesh, calib):
    r = mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    if not r.multi_face_landmarks: return None, calib
    pts = [(p.x*224, p.y*224) for p in r.multi_face_landmarks[0].landmark]
    def eye_center(pts, out_idx, in_idx, up_idx, dn_idx):
        return ((pts[out_idx][0]+pts[in_idx][0])/2.0, (pts[up_idx][1]+pts[dn_idx][1])/2.0)
    eye_c_l = eye_center(pts, L_OUT, L_IN, L_UP, L_DN)
    eye_c_r = eye_center(pts, R_OUT, R_IN, R_UP, R_DN)
    ipd = dist(eye_c_l, eye_c_r)
    ear_l = EAR(pts, L_UP, L_DN, L_OUT, L_IN)
    ear_r = EAR(pts, R_UP, R_DN, R_OUT, R_IN)
    mar = MAR(pts)
    face_area = 224*224
    B_raw = brow_raise_B(pts, eye_c_l, eye_c_r)
    M_raw = mouth_curve_M(pts)
    def upd(key, v):
        mm = calib.setdefault(key, {"min": float(v), "max": float(v)})
        mm["min"] = float(min(mm["min"], v)); mm["max"] = float(max(mm["max"], v))
    for k,v in [("face_area",face_area),("ipd",ipd),("mar",mar),("B_raw",B_raw),("M_raw",M_raw)]: upd(k,v)
    Aw = 1.0 - minmax_norm(face_area, calib["face_area"]["min"], calib["face_area"]["max"])
    Ca =       minmax_norm(ipd,       calib["ipd"]["min"],       calib["ipd"]["max"])
    Hp =       minmax_norm(M_raw,     calib["M_raw"]["min"],     calib["M_raw"]["max"])
    def center_pm1(x, lo, hi):
        if hi-lo<1e-6: return 0.0
        return float(np.clip((x-(lo+hi)/2)/((hi-lo)/2), -1, 1))
    B = center_pm1(B_raw, calib["B_raw"]["min"], calib["B_raw"]["max"])
    M = center_pm1(M_raw, calib["M_raw"]["min"], calib["M_raw"]["max"])
    mag = (B**2 + M**2)**0.5
    ang = np.arctan2(B, M)
    feats = {"Aw":Aw,"Ca":Ca,"Hp":Hp,"B":B,"M":M,"magnitude":mag,"direction":ang,
             "ear_l":ear_l,"ear_r":ear_r,"mar":mar,"ipd":ipd,"face_area":face_area}
    return feats, calib

def main():
    bundle = joblib.load("data/tabular_baseline.pkl")
    scaler, clf, FEATS = bundle["scaler"], bundle["clf"], bundle["feats"]

    mp_det = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

    cap = cv2.VideoCapture(0); calib={}
    while True:
        ok, frame = cap.read()
        if not ok: break
        h,w = frame.shape[:2]
        det = mp_det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if det.detections:
            d = det.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(d.xmin*w)); y1 = max(0, int(d.ymin*h))
            ww = int(d.width*w); hh = int(d.height*h)
            x2 = min(w-1, x1+ww); y2 = min(h-1, y1+hh)
            face = frame[y1:y2, x1:x2].copy()
            # 簡單兩眼估計（沿用 collect_and_label）
            r1 = mp_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            if r1.multi_face_landmarks:
                pts = [(p.x*face.shape[1], p.y*face.shape[0]) for p in r1.multi_face_landmarks[0].landmark]
                left_eye  = (pts[L_OUT][0], pts[L_UP][1])
                right_eye = (pts[R_OUT][0], pts[R_UP][1])
                aligned = affine_align_by_eyes(face, left_eye, right_eye, out_size=224)
                feats, calib = compute_feats_for_frame(aligned, mp_mesh, calib)
                if feats:
                    x = np.array([[feats[k] for k in FEATS]], dtype=float)
                    xs = scaler.transform(x)
                    p = clf.predict_proba(xs)[0][1]
                    pred = "困惑" if p>=0.5 else "理解"
                    frame = put_text(frame, f"{pred}  conf:{p:.2f}", (x1, y1-10))
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)

        cv2.imshow("tabular-infer", frame)
        if (cv2.waitKey(1)&0xFF)==27: break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
