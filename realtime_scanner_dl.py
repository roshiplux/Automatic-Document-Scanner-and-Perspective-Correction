import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(16, 32, 64)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        rev_features = list(features[::-1])
        up_ch = features[-1] * 2
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(up_ch, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(up_ch, feat))
            up_ch = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)
        return self.final_conv(x)


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)



def four_point_transform(image: np.ndarray, pts: np.ndarray):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(max(width_a, width_b)), 1)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(max(height_a, height_b)), 1)
    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped, rect



def scanner_enhance(warped: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10)



def predict_mask(model, frame_bgr, image_size=256, threshold=0.5):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = rgb.shape[:2]
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (probs > threshold).astype(np.uint8) * 255
    mask_big = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_big



def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)



def contour_to_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)



def stack_views(frame, mask, warped, scan):
    h, w = frame.shape[:2]
    if mask is None:
        mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack([frame, cv2.resize(mask_vis, (w, h))])

    if warped is None:
        warped_vis = np.zeros_like(frame)
    else:
        warped_vis = cv2.resize(warped, (w, h))

    if scan is None:
        scan_vis = np.zeros_like(frame)
    else:
        scan_vis = cv2.cvtColor(cv2.resize(scan, (w, h)), cv2.COLOR_GRAY2BGR)

    bottom = np.hstack([warped_vis, scan_vis])
    return np.vstack([top, bottom])



def main():
    parser = argparse.ArgumentParser(description="Real-time document scanner using your trained TinyUNet.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index, usually 0 or 1")
    parser.add_argument("--model", type=str, default="outputs/best_tiny_unet.pth", help="Path to trained weights")
    parser.add_argument("--image_size", type=int, default=256, help="Model input size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--infer_every", type=int, default=2, help="Run model every N frames")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--save_dir", type=str, default="outputs/realtime_dl", help="Folder to save scans")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TinyUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    print(f"Using device: {DEVICE}")
    print("Press q to quit, s to save current warped page and scan.")

    frame_id = 0
    cached_mask = None
    last_warped = None
    last_scan = None
    fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_id % max(args.infer_every, 1) == 0 or cached_mask is None:
            cached_mask = predict_mask(model, frame, image_size=args.image_size, threshold=args.threshold)
        frame_id += 1

        display = frame.copy()
        warped = None
        scan = None
        contour = largest_contour(cached_mask)

        if contour is not None:
            quad = contour_to_quad(contour)
            warped, ordered = four_point_transform(frame, quad)
            scan = scanner_enhance(warped)
            cv2.polylines(display, [ordered.astype(np.int32)], True, (0, 255, 0), 3)
            last_warped = warped.copy()
            last_scan = scan.copy()

        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            fps_value = fps_counter / (now - fps_time)
            fps_counter = 0
            fps_time = now

        cv2.putText(display, f"FPS: {fps_value:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        board = stack_views(display, cached_mask, warped, scan)
        cv2.imshow("Real-time Document Scanner (DL-assisted)", board)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s") and last_warped is not None and last_scan is not None:
            ts = int(time.time())
            warped_path = save_dir / f"warped_{ts}.jpg"
            scan_path = save_dir / f"scan_{ts}.jpg"
            cv2.imwrite(str(warped_path), last_warped)
            cv2.imwrite(str(scan_path), last_scan)
            print(f"Saved {warped_path} and {scan_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
