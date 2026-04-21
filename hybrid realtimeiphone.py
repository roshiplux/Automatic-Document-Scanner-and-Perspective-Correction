import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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


def open_camera(camera_index: int, width: int | None = None, height: int | None = None):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def probe_cameras(max_index: int = 6):
    found = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if ok:
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append(idx)
        cap.release()
    return found


def classical_preprocess(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, clahe, blur, edges, closed


def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_quad(contour):
    if contour is None:
        return None
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    rect = cv2.minAreaRect(contour)
    return cv2.boxPoints(rect).astype(np.float32)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def four_point_transform(image_bgr, pts):
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
    warped = cv2.warpPerspective(image_bgr, matrix, (max_width, max_height))
    return warped, rect


def scanner_enhance(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    scan = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        10,
    )
    return scan


def classical_candidate_mask(frame_bgr):
    gray, clahe, blur, edges, closed = classical_preprocess(frame_bgr)
    contour = largest_contour(closed)
    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    debug = {
        "gray": gray,
        "clahe": clahe,
        "blur": blur,
        "edges": edges,
        "closed": closed,
    }
    return mask, debug


def predict_dl_mask(model, frame_bgr, image_size, device, threshold):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask_small = (probs > threshold).astype(np.uint8) * 255
    mask_big = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_big


def smooth_quad(prev_quad, current_quad, alpha=0.8):
    if prev_quad is None:
        return current_quad
    return alpha * prev_quad + (1.0 - alpha) * current_quad


def stack_small(images, labels, scale=0.4):
    smalls = []
    for img, label in zip(images, labels):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        smalls.append(img)
    return np.hstack(smalls)


def main():
    parser = argparse.ArgumentParser(description="Hybrid real-time document scanner for webcam or iPhone camera")
    parser.add_argument("--camera", type=int, default=-1, help="Camera index. Use -1 to auto-probe.")
    parser.add_argument("--max-probe", type=int, default=6, help="Highest camera index to probe when --camera=-1")
    parser.add_argument("--model", type=str, default="outputs/best_tiny_unet.pth")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-dir", type=str, default="outputs")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mirror", action="store_true", help="Mirror preview horizontally")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TinyUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if args.camera == -1:
        found = probe_cameras(args.max_probe)
        print(f"Available cameras: {found}")
        if not found:
            raise RuntimeError("No camera found. On Mac, make sure Continuity Camera is enabled on iPhone.")
        camera_index = found[-1]
    else:
        camera_index = args.camera

    cap = open_camera(camera_index, args.width, args.height)
    if cap is None:
        raise RuntimeError(f"Could not open camera index {camera_index}")

    print(f"Using camera index: {camera_index}")
    print("Keys: q quit | s save current result | d toggle debug windows")

    prev_quad = None
    show_debug = True
    save_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        classical_mask, debug = classical_candidate_mask(frame)
        dl_mask = predict_dl_mask(model, frame, args.image_size, device, args.threshold)

        if cv2.countNonZero(classical_mask) > 0:
            combined = cv2.bitwise_and(dl_mask, classical_mask)
            if cv2.countNonZero(combined) < 0.05 * max(cv2.countNonZero(dl_mask), 1):
                combined = dl_mask
        else:
            combined = dl_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

        contour = largest_contour(combined)
        quad = contour_to_quad(contour)

        overlay = frame.copy()
        warped = None
        scan = None

        if quad is not None:
            quad = order_points(quad)
            quad = smooth_quad(prev_quad, quad, alpha=0.75)
            prev_quad = quad.copy()
            warped, quad = four_point_transform(frame, quad)
            scan = scanner_enhance(warped)
            cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 0), 3)
            cv2.putText(overlay, "Hybrid DL + CV", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(overlay, "No page detected", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("1 - Live overlay", overlay)
        cv2.imshow("2 - DL mask", dl_mask)
        cv2.imshow("3 - Combined mask", combined)

        if warped is not None:
            cv2.imshow("4 - Warped", warped)
        if scan is not None:
            cv2.imshow("5 - Scanner output", scan)
        if show_debug:
            strip = stack_small(
                [debug["gray"], debug["clahe"], debug["blur"], debug["edges"], debug["closed"]],
                ["gray", "clahe", "blur", "canny", "morph"],
            )
            cv2.imshow("0 - Preprocessing", strip)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("0 - Preprocessing")
        if key == ord("s") and warped is not None and scan is not None:
            base = save_dir / f"iphone_scan_{save_index:03d}"
            cv2.imwrite(str(base.with_name(base.name + "_overlay.jpg")), overlay)
            cv2.imwrite(str(base.with_name(base.name + "_warped.jpg")), warped)
            cv2.imwrite(str(base.with_name(base.name + "_scan.jpg")), scan)
            print(f"Saved set {save_index} to {save_dir}")
            save_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
