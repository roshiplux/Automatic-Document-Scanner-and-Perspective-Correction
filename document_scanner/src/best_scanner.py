#!/usr/bin/env python3
"""
BEST Document Scanner - Optimized for Real-World Use
Simple, Fast, Reliable - No unnecessary complexity
"""

import cv2
import numpy as np
import os
from datetime import datetime
from simple_detector import SimpleDocumentDetector
from perspective_corrector import PerspectiveCorrector


def main():
    """Run the best document scanner."""
    print("\n" + "="*70)
    print("  🎯 BEST DOCUMENT SCANNER - Simple & Reliable")
    print("="*70)
    
    # Choose enhancement mode
    print("\n📝 Choose Enhancement Mode:")
    print("  1. Original (No enhancement)")
    print("  2. Color Enhanced")
    print("  3. Black & White Adaptive (Best for text)")
    print("  4. Black & White Otsu (High contrast)")
    
    while True:
        choice = input("\nEnter choice (1-4) [default: 3]: ").strip()
        if not choice:
            choice = "3"
        
        if choice in ["1", "2", "3", "4"]:
            mode_map = {
                "1": ("none", "Original"),
                "2": ("color", "Color Enhanced"),
                "3": ("adaptive", "Black & White Adaptive"),
                "4": ("otsu", "Black & White Otsu")
            }
            enhancement_mode, mode_name = mode_map[choice]
            print(f"\n✅ Selected: {mode_name}\n")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    # Detect cameras (SKIP Camera 0 - built-in laptop camera)
    print("\n🔍 Detecting mobile cameras (built-in camera disabled)...")
    cameras = []
    for i in range(1, 4):  # Start from 1, skip 0 (built-in camera)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append((i, w, h))
                print(f"  ✅ Mobile Camera {i}: {w}x{h}")
            cap.release()
    
    if not cameras:
        print("\n❌ No mobile camera found!")
        print("\n📱 Please connect your mobile phone camera:")
        print("\n  iPhone Users:")
        print("    • Use Continuity Camera (iOS 16+)")
        print("    • Or install: EpocCam, iVCam, or similar")
        print("\n  Android Users:")
        print("    • Install: DroidCam, IP Webcam, or similar")
        print("    • Download companion app on Mac")
        print("\n  Connect via USB or WiFi, then try again.\n")
        return
    
    # Use first available mobile camera
    camera_idx = cameras[0][0]
    print(f"\n🎉 Using Mobile Camera {camera_idx}\n")
    
    # Initialize
    detector = SimpleDocumentDetector()
    corrector = PerspectiveCorrector()
    
    print("="*70)
    print("  📋 INSTRUCTIONS")
    print("="*70)
    print("\n1. Place document on DARK/CONTRASTING surface")
    print("2. Ensure GOOD LIGHTING (no shadows)")
    print("3. Keep document FLAT and fully visible")
    print("4. Hold camera STEADY above document")
    print("\n⌨️  CONTROLS:")
    print("  SPACE - Capture and scan")
    print("  'c'   - Capture frame (freeze)")
    print("  'r'   - Resume live view")
    print("  'q'   - Quit")
    print("="*70 + "\n")
    
    # Open camera
    cap = cv2.VideoCapture(camera_idx)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    
    frozen_frame = None
    frozen_result = None
    
    print("📸 Camera started - Position your document...\n")
    
    while True:
        if frozen_frame is None:
            # Live view
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Detect document
            success, corners, vis = detector.detect(frame)
            
            # Add status text
            if success:
                cv2.putText(vis, "DOCUMENT DETECTED - Press SPACE to scan", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis, "or press 'c' to freeze frame", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv2.putText(vis, "Position document - Green box will appear", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                cv2.putText(vis, "Use DARK surface under paper!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            cv2.putText(vis, "Press 'q' to quit", 
                       (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Document Scanner', vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Freeze frame
                frozen_frame = frame.copy()
                frozen_result = (success, corners, vis.copy())
                print("📸 Frame frozen - adjust if needed, press SPACE to scan or 'r' to resume")
            elif key == ord(' ') and success:
                # Immediate scan
                print("\n🔄 Scanning...")
                save_scan(frame, corners, corrector, enhancement_mode, mode_name)
                
        else:
            # Frozen frame mode
            success, corners, vis = frozen_result
            
            cv2.putText(vis, "FROZEN - Press SPACE to scan, 'r' to resume", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow('Document Scanner', vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Resume live view
                frozen_frame = None
                frozen_result = None
                print("▶️  Resumed live view")
            elif key == ord(' ') and success:
                # Scan frozen frame
                print("\n🔄 Scanning...")
                save_scan(frozen_frame, corners, corrector, enhancement_mode, mode_name)
                frozen_frame = None
                frozen_result = None
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Scanner closed\n")


def save_scan(image, corners, corrector, enhancement_mode, mode_name):
    """Save scanned document and show preview."""
    # Apply perspective correction
    corrected = corrector.apply_perspective_transform(image, corners)
    
    # Auto-rotate if document is vertical but saved horizontally
    h, w = corrected.shape[:2]
    
    # Calculate aspect ratio of detected document corners
    # If corners suggest vertical orientation but result is horizontal, rotate
    if corners is not None and len(corners) == 4:
        # Calculate distances between corner points
        pts = corners.reshape(4, 2)
        # Distance between top-left and top-right (width)
        width_dist = np.linalg.norm(pts[0] - pts[1])
        # Distance between top-left and bottom-left (height)
        height_dist = np.linalg.norm(pts[0] - pts[3])
        
        # If original detection was taller than wide, ensure output is portrait
        if height_dist > width_dist and w > h:
            # Rotate 90 degrees clockwise to make it vertical
            corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
            print("   🔄 Auto-rotated to portrait orientation")
    
    # Enhance based on selected mode
    if enhancement_mode == 'none':
        enhanced = corrected
    else:
        enhanced = corrector.enhance_document(corrected, method=enhancement_mode)
    
    # Get absolute path to output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"scan_{timestamp}.png")
    cv2.imwrite(filename, enhanced)
    
    print(f"✅ Saved: {filename}")
    print(f"   📄 {mode_name} applied")
    
    # Show preview in new window
    preview = enhanced.copy()
    
    # Resize for preview (max 1000 width for better display)
    h, w = preview.shape[:2]
    if w > 1000:
        scale = 1000 / w
        new_w, new_h = int(w * scale), int(h * scale)
        preview = cv2.resize(preview, (new_w, new_h))
    
    # Add text overlay
    if len(preview.shape) == 3:  # Color image
        color = (0, 255, 0)
    else:  # Grayscale
        color = 255
    
    cv2.putText(preview, f"SAVED: {mode_name}", 
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(preview, "Preview closes in 3 seconds or press any key", 
               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Show preview window
    cv2.imshow('Scanned Document - Preview', preview)
    cv2.waitKey(3000)  # Auto-close after 3 seconds OR on any key press
    cv2.destroyWindow('Scanned Document - Preview')
    
    # Open output folder in Finder
    os.system(f'open "{output_dir}"')
    print()



if __name__ == "__main__":
    main()
