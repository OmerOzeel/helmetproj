import cv2
import numpy as np
import platform
import time


MODEL_PATH = r"C:\Users\OZEL MSI\Desktop\helmet\helmet\best.onnx"
VIDEO_PATH = r"C:\Users\OZEL MSI\Desktop\helmet\helmet\helmet_video.mp4"
IMG_SIZE = 640
CONF_TH = 0.25
IOU_TH = 0.45
HELMET_ID = 1  
apply_sigmoid = False
fps_limit = 5

def letterbox(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = new_shape / max(h, w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    img_resize = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img_pad = cv2.copyMakeBorder(img_resize, dh, dh, dw, dw,
                                 cv2.BORDER_CONSTANT, value=color)
    return img_pad, r, dw, dh

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# MODEL 
net = cv2.dnn.readNetFromONNX(MODEL_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#  VİDEO 
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise IOError(f"‼️  Video/Kamera açılamadı: {VIDEO_PATH}")

cv2.namedWindow("Baret Tespiti", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Baret Tespiti", 640, 480)


prev_time = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    curr_time = time.time()
    if curr_time - prev_time < 1 / fps_limit:
        continue
    prev_time = curr_time

    img, r, dw, dh = letterbox(frame)
    blob = cv2.dnn.blobFromImage(img, 1/255., (IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    pred = net.forward()[0]

    if apply_sigmoid:
        pred[:, 4:] = sigmoid(pred[:, 4:])

    boxes, scores = [], []
    H0, W0 = frame.shape[:2]

    for det in pred:
        cx, cy, w, h = det[:4]
        obj_conf = det[4]
        cls_scores = det[5:]
        class_id = int(np.argmax(cls_scores))
        conf = obj_conf * cls_scores[class_id]

        if conf < CONF_TH or class_id != HELMET_ID:
            continue

        x1 = (cx - w/2) - dw
        y1 = (cy - h/2) - dh
        x2 = (cx + w/2) - dw
        y2 = (cy + h/2) - dh
        x1, y1, x2, y2 = [int(v / r) for v in (x1, y1, x2, y2)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W0 - 1, x2), min(H0 - 1, y2)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

    idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_TH, IOU_TH)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Helmet {scores[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Baret Tespiti", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
