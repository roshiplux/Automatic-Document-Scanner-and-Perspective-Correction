"""
Phone Camera Live Scanner with Movement Detection
Enhanced version with real-time angle and stability guidance
"""

import cv2
import numpy as np
from document_detector import DocumentDetector
from perspective_corrector import PerspectiveCorrector


def draw_camera_guidance(frame, corners_detected=False, stable=False, movement_amount=0):
    """Draw real-time guidance overlay for optimal camera positioning."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw center crosshair
    center_x, center_y = w // 2, h // 2
    cv2.line(overlay, (center_x - 40, center_y), (center_x + 40, center_y), (255, 255, 255), 2)
    cv2.line(overlay, (center_x, center_y - 40), (center_x, center_y + 40), (255, 255, 255), 2)
    
    # Draw target rectangle for document placement
    margin = int(min(w, h) * 0.1)
    cv2.rectangle(overlay, (margin, margin), (w - margin, h - margin), (150, 150, 150), 2)
    
    # Status indicator box
    if corners_detected and stable:
        status_text = "✓ PERFECT - PRESS SPACE"
        status_color = (0, 255, 0)
        box_color = (0, 200, 0)
    elif corners_detected:
        status_text = "HOLD STEADY"
        status_color = (0, 255, 255)
        box_color = (0, 200, 200)
    else:
        status_text = "POSITION DOCUMENT"
        status_color = (0, 165, 255)
        box_color = (0, 130, 200)
    
    # Draw status box at top
    cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.rectangle(overlay, (10, 10), (400, 80), box_color, 3)
    cv2.putText(overlay, status_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    # Movement indicator
    if corners_detected:
        if movement_amount < 15:
            move_text = "Steady!"
            move_color = (0, 255, 0)
        elif movement_amount < 30:
            move_text = "Small movement"
            move_color = (0, 255, 255)
        else:
            move_text = "Moving"
            move_color = (0, 165, 255)
        cv2.putText(overlay, move_text, (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, move_color, 2)
    
    # Draw guidance tips at bottom
    tips = [
        "📱 Hold phone DIRECTLY above (90°)",
        "📐 Keep parallel to document",
        "💡 Avoid shadows & glare",
        "🎯 Center document in frame"
    ]
    
    y_offset = h - 130
    cv2.rectangle(overlay, (5, y_offset - 35), (w - 5, h - 5), (0, 0, 0), -1)
    for i, tip in enumerate(tips):
        cv2.putText(overlay, tip, (15, y_offset + i * 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Blend overlay
    return cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)


def phone_camera_scan(camera_index):
    """Live phone camera scanner with movement and angle detection."""
    # Use very relaxed detection for live mode
    detector = DocumentDetector(min_area_ratio=0.01, max_area_ratio=0.98)
    corrector = PerspectiveCorrector()
    
    print("\n" + "="*70)
    print("  📱 PHONE CAMERA LIVE SCANNER")
    print("="*70)
    print("\n📐 Camera Positioning Guide:")
    print("  ✓ Hold phone DIRECTLY above document (90° angle)")
    print("  ✓ Keep phone parallel to document surface")
    print("  ✓ Distance: 30-40cm above document")
    print("  ✓ Center document in frame")
    print("  ✓ Avoid shadows from your hand")
    print("  ✓ Use steady hands or stand/tripod")
    print("  ✓ Place document on dark/contrasting surface")
    print("\n🎯 Wait for '✓ PERFECT' message before capturing!")
    print("\n⌨️  Controls:")
    print("  SPACE - Capture when stable")
    print("  'q'   - Quit")
    print("="*70 + "\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Ultra-lenient movement tracking
    prev_corners = None
    stable_frames = 0
    required_stable_frames = 2  # Only need 2 stable frames (was 3)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Failed to capture frame")
            break
        
        # Detect document with more lenient temporal smoothing
        success, corners, vis_image = detector.detect(frame, use_temporal_smoothing=True)
        
        # Calculate camera movement
        movement_amount = 0
        is_stable = False
        
        if success and corners is not None:
            if prev_corners is not None:
                # Calculate average movement
                movement_amount = np.mean(np.linalg.norm(corners - prev_corners, axis=1))
                
                if movement_amount < 20:  # Ultra-lenient: less than 20 pixels (was 12)
                    stable_frames += 1
                else:
                    stable_frames = max(0, stable_frames - 0.5)  # Very slow decay
                    stable_frames = max(0, stable_frames - 2)
            
            is_stable = stable_frames >= required_stable_frames
            prev_corners = corners.copy()
        else:
            stable_frames = 0
            prev_corners = None
        
        # Add guidance overlay
        vis_image = draw_camera_guidance(vis_image, success, is_stable, movement_amount)
        
        # Show live view
        cv2.imshow('📱 Phone Camera Scanner [SPACE=Capture | Q=Quit]', vis_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            if success and is_stable:
                print("\n✅ Document captured perfectly!")
                print(f"   📊 Movement: {movement_amount:.1f}px | Stable frames: {stable_frames}")
                
                # Apply perspective correction
                corrected = corrector.apply_perspective_transform(frame, corners)
                
                # Show enhancement options
                print("\nSelect enhancement mode:")
                print("  1 - Original Color")
                print("  2 - Enhanced Color")
                print("  3 - Black & White Adaptive (Best for notes)")
                print("  4 - Black & White Otsu")
                
                # For now, auto-apply mode 3
                enhanced = corrector.enhance_document(corrected, method='adaptive')
                
                # Save
                import os
                from datetime import datetime
                output_dir = "../output"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scan_{timestamp}.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, enhanced)
                print(f"\n💾 Saved: {output_path}\n")
                
            elif success:
                print(f"⚠️  Camera moving ({movement_amount:.1f}px) - Hold steady!")
                print(f"   Need {required_stable_frames - stable_frames} more stable frames")
            else:
                print("⚠️  No document detected")
                print("   • Check camera angle (straight down)")
                print("   • Improve lighting")
                print("   • Use dark background")
                print("   • Ensure full document visible")
                
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("  � PHONE CAMERA LIVE SCANNER")
    print("="*70)
    print("\n🔍 Scanning for ALL available cameras...\n")
    
    # Scan ALL camera indices (0-5) to find iPhone/phone cameras
    available = []
    for i in range(0, 6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                # Determine camera type
                if i == 0:
                    cam_name = "Built-in Camera"
                else:
                    cam_name = "External Camera (iPhone/Android)"
                
                available.append((i, w, h, fps, cam_name))
                print(f"✅ Camera {i}: {cam_name} - {w}x{h} @ {fps}fps")
            cap.release()
    
    if not available:
        print("\n❌ NO CAMERAS FOUND!")
        print("\n📱 For iPhone:")
        print("  1. Ensure iPhone unlocked near Mac")
        print("  2. Check WiFi/Bluetooth enabled")
        print("  3. Same Apple ID on both devices")
        print("\n🤖 For Android:")
        print("  1. Install DroidCam app")
        print("  2. Connect via USB")
        print("\n⚠️  Try restarting your computer\n")
        sys.exit(1)
    
    # If multiple cameras, prefer non-zero index (external)
    if len(available) > 1:
        # Use Camera 1 or higher if available (external/iPhone)
        camera_idx = available[1][0] if len(available) > 1 else available[0][0]
        print(f"\n✨ Preferring external camera: Camera {camera_idx}")
    else:
        camera_idx = available[0][0]
    
    print(f"🎉 Using Camera {camera_idx} - {available[camera_idx if camera_idx < len(available) else 0][4]}\n")
    
    phone_camera_scan(camera_idx)
