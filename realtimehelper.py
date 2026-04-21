import cv2

for idx in range(6):
    cap = cv2.VideoCapture(idx)
    ok, frame = cap.read()
    if ok:
        h, w = frame.shape[:2]
        print(f"Camera index {idx}: available ({w}x{h})")
    else:
        print(f"Camera index {idx}: not available")
    cap.release()