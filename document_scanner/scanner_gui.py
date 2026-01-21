#!/usr/bin/env python3
"""
📸 Document Scanner - Desktop GUI
Simple, Fast, Professional - Mobile Camera Only
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import sys
from datetime import datetime
import threading

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simple_detector import SimpleDocumentDetector
from perspective_corrector import PerspectiveCorrector


class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📸 Document Scanner")
        self.root.geometry("1400x900")
        self.root.configure(bg="#34495e")
        
        # Initialize
        self.detector = SimpleDocumentDetector()
        self.corrector = PerspectiveCorrector()
        self.output_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Variables
        self.current_image = None
        self.current_corners = None
        self.enhancement_mode = tk.StringVar(value="adaptive")
        self.camera_running = False
        self.cap = None
        self.rotation_angle = 0
        self.auto_rotate = tk.BooleanVar(value=True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title_frame = tk.Frame(self.root, bg='#34495e', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="📸 Document Scanner", 
            font=("Helvetica", 32, "bold"),
            bg='white',
            fg='#34495e'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='white')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='#34495e', width=300, relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        self.setup_controls(left_panel)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.setup_image_display(right_panel)
    
    def setup_controls(self, parent):
        """Setup control buttons and settings"""
        # Buttons section
        btn_frame = tk.Frame(parent, bg='#34495e')
        btn_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Upload button
        upload_btn = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            command=self.upload_image,
            font=("Helvetica", 14, "bold"),
            bg='#1f77b4',
            fg='white',
            height=2,
            cursor='hand2'
        )
        upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Camera button
        self.camera_btn = tk.Button(
            btn_frame,
            text="📹 Start Camera",
            command=self.toggle_camera,
            font=("Helvetica", 14, "bold"),
            bg='#28a745',
            fg='white',
            height=2,
            cursor='hand2'
        )
        self.camera_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Scan button
        self.scan_btn = tk.Button(
            btn_frame,
            text="🚀 Scan Document",
            command=self.scan_document,
            font=("Helvetica", 14, "bold"),
            bg='#ff6b6b',
            fg='white',
            height=2,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.scan_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Open folder button
        folder_btn = tk.Button(
            btn_frame,
            text="📂 Open Output Folder",
            command=self.open_output_folder,
            font=("Helvetica", 12),
            bg='#6c757d',
            fg='white',
            height=2,
            cursor='hand2'
        )
        folder_btn.pack(fill=tk.X)
        
        # Settings section
        settings_frame = tk.LabelFrame(
            parent, 
            text="⚙️ Settings",
            font=("Helvetica", 14, "bold"),
            bg='white',
            padx=20,
            pady=10
        )
        settings_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(
            settings_frame,
            text="Enhancement Mode:",
            font=("Helvetica", 11),
            bg='white'
        ).pack(anchor=tk.W, pady=(0, 5))
        
        modes = [
            ("📷 Original", "none"),
            ("🎨 Color Enhanced", "color"),
            ("📝 Adaptive (Best for text)", "adaptive"),
            ("⚫ Otsu (High contrast)", "otsu")
        ]
        
        for text, value in modes:
            rb = tk.Radiobutton(
                settings_frame,
                text=text,
                variable=self.enhancement_mode,
                value=value,
                font=("Helvetica", 10),
                bg='white',
                cursor='hand2'
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Rotation control
        tk.Label(
            settings_frame,
            text="\nCamera Rotation:",
            font=("Helvetica", 11, "bold"),
            bg='white'
        ).pack(anchor=tk.W, pady=(10, 5))
        
        auto_rotate_cb = tk.Checkbutton(
            settings_frame,
            text="🔄 Auto-rotate (recommended)",
            variable=self.auto_rotate,
            font=("Helvetica", 10),
            bg='white',
            cursor='hand2'
        )
        auto_rotate_cb.pack(anchor=tk.W, pady=2)
        
        # Manual rotation buttons
        rotation_btn_frame = tk.Frame(settings_frame, bg='white')
        rotation_btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(
            rotation_btn_frame,
            text="↶ 90°",
            command=lambda: self.rotate_manual(-90),
            font=("Helvetica", 9),
            width=6
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            rotation_btn_frame,
            text="↷ 90°",
            command=lambda: self.rotate_manual(90),
            font=("Helvetica", 9),
            width=6
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            rotation_btn_frame,
            text="↻ 180°",
            command=lambda: self.rotate_manual(180),
            font=("Helvetica", 9),
            width=6
        ).pack(side=tk.LEFT, padx=2)
        
        # Instructions section
        inst_frame = tk.LabelFrame(
            parent,
            text="📋 Instructions",
            font=("Helvetica", 14, "bold"),
            bg='white',
            padx=20,
            pady=10
        )
        inst_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        instructions = """
📱 MOBILE CAMERA REQUIRED
(Built-in laptop camera is disabled)

Connect your mobile camera:

iPhone:
• Continuity Camera (iOS 16+) OR
• EpocCam / iVCam app

Android:
• DroidCam / IP Webcam app

Then:
1. Click 'Start Camera'
2. Place document on dark surface
3. Ensure good lighting
4. Keep document flat & visible
5. Click 'Scan' when green box appears

💡 Tips:
• Use Adaptive for handwritten notes
• Use Otsu for printed documents
• Dark background improves detection
        """
        
        inst_label = tk.Label(
            inst_frame,
            text=instructions,
            font=("Helvetica", 10),
            bg='white',
            justify=tk.LEFT
        )
        inst_label.pack(anchor=tk.W)
    
    def setup_image_display(self, parent):
        """Setup image display area"""
        # Status label
        self.status_label = tk.Label(
            parent,
            text="No image loaded",
            font=("Helvetica", 14),
            bg='white',
            fg='#666'
        )
        self.status_label.pack(pady=20)
        
        # Image canvas
        self.canvas = tk.Label(parent, bg='#e0e0e0')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
    
    def upload_image(self):
        """Upload and process an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Document Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.heic"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.stop_camera()
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                self.process_image()
            else:
                messagebox.showerror("Error", "Failed to load image")
    
    def toggle_camera(self):
        """Start or stop camera"""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera feed - MOBILE CAMERA ONLY"""
        # Reset rotation for new camera session
        self.rotation_angle = 0
        self.auto_rotate.set(True)
        
        # Block built-in camera (Camera 0) - Only allow Camera 1+ (mobile/external)
        camera_found = False
        
        for cam_idx in [1, 2, 3]:  # Skip Camera 0 (built-in laptop camera)
            self.cap = cv2.VideoCapture(cam_idx)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    camera_found = True
                    self.status_label.config(
                        text=f"📱 Mobile Camera {cam_idx} connected - Position your document",
                        fg='#28a745'
                    )
                    break
                self.cap.release()
        
        if not camera_found:
            messagebox.showinfo(
                "📱 Connect Mobile Camera",
                "Built-in laptop camera is disabled.\n\n"
                "Please connect your mobile phone camera:\n\n"
                "📱 iPhone Users:\n"
                "1. Install 'Continuity Camera' (iOS 16+)\n"
                "2. Or use apps like 'EpocCam' or 'iVCam'\n"
                "3. Connect via USB or WiFi\n\n"
                "📱 Android Users:\n"
                "1. Install 'DroidCam' or 'IP Webcam'\n"
                "2. Download companion app on Mac\n"
                "3. Connect via USB or WiFi\n\n"
                "Then click 'Start Camera' again."
            )
            return
        
        self.camera_running = True
        self.camera_btn.config(text="⏹ Stop Camera", bg='#dc3545')
        
        self.update_camera()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="📹 Start Camera", bg='#28a745')
    
    def update_camera(self):
        """Update camera feed"""
        if not self.camera_running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Apply rotation correction
            frame = self.apply_rotation(frame)
            self.current_image = frame.copy()
            self.process_image()
        
        if self.camera_running:
            self.root.after(100, self.update_camera)
    
    def apply_rotation(self, image):
        """Apply rotation to correct camera orientation"""
        if not self.auto_rotate.get() and self.rotation_angle == 0:
            return image
        
        # Auto-detect rotation if enabled
        if self.auto_rotate.get() and self.rotation_angle == 0:
            h, w = image.shape[:2]
            # If width > height significantly, phone is likely held vertically
            # but streaming in landscape - rotate 90 degrees clockwise
            if w > h * 1.3:  # Landscape ratio
                self.rotation_angle = 90
        
        # Apply rotation
        if self.rotation_angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif self.rotation_angle == 270 or self.rotation_angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return image
    
    def rotate_manual(self, angle):
        """Manually rotate camera view"""
        self.auto_rotate.set(False)  # Disable auto-rotation
        self.rotation_angle = (self.rotation_angle + angle) % 360
        if self.current_image is not None:
            self.process_image()
    
    def process_image(self):
        """Detect document in image"""
        if self.current_image is None:
            return
        
        # For uploaded images, apply rotation if needed
        image_to_process = self.current_image
        if not self.camera_running:
            image_to_process = self.apply_rotation(self.current_image)
        
        # Detect document
        success, corners, vis = self.detector.detect(image_to_process)
        
        self.current_corners = corners if success else None
        
        # Update status
        if success:
            self.status_label.config(
                text="✅ Document detected - Click 'Scan' to process",
                fg='#28a745'
            )
            self.scan_btn.config(state=tk.NORMAL)
        else:
            self.status_label.config(
                text="⚠️ No document detected - Adjust position or lighting",
                fg='#ffc107'
            )
            self.scan_btn.config(state=tk.DISABLED)
        
        # Display image
        self.display_image(vis)
    
    def display_image(self, image):
        """Display image on canvas"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.canvas.winfo_width() - 40
        canvas_height = self.canvas.winfo_height() - 40
        
        if canvas_width > 100 and canvas_height > 100:  # Canvas initialized
            h, w = image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Convert to PhotoImage
            img_pil = Image.fromarray(resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.canvas.config(image=img_tk)
            self.canvas.image = img_tk
    
    def scan_document(self):
        """Scan the detected document"""
        if self.current_image is None or self.current_corners is None:
            return
        
        self.status_label.config(text="🔄 Scanning...", fg='#1f77b4')
        self.root.update()
        
        # Stop camera if running
        was_running = self.camera_running
        if was_running:
            self.stop_camera()
        
        # Apply perspective correction
        corrected = self.corrector.apply_perspective_transform(
            self.current_image, 
            self.current_corners
        )
        
        # Auto-rotate if needed
        h, w = corrected.shape[:2]
        if self.current_corners is not None and len(self.current_corners) == 4:
            pts = self.current_corners.reshape(4, 2)
            width_dist = np.linalg.norm(pts[0] - pts[1])
            height_dist = np.linalg.norm(pts[0] - pts[3])
            
            if height_dist > width_dist and w > h:
                corrected = cv2.rotate(corrected, cv2.ROTATE_90_CLOCKWISE)
        
        # Enhance
        mode = self.enhancement_mode.get()
        if mode == 'none':
            enhanced = corrected
        else:
            enhanced = self.corrector.enhance_document(corrected, method=mode)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"scan_{timestamp}.png")
        cv2.imwrite(filename, enhanced)
        
        # Show success message
        mode_names = {
            'none': 'Original',
            'color': 'Color Enhanced',
            'adaptive': 'Black & White Adaptive',
            'otsu': 'Black & White Otsu'
        }
        
        messagebox.showinfo(
            "Success",
            f"✅ Document scanned successfully!\n\n"
            f"Mode: {mode_names[mode]}\n"
            f"Saved: {filename}"
        )
        
        # Display result
        self.status_label.config(
            text=f"✅ Saved: {filename}",
            fg='#28a745'
        )
        self.display_image(enhanced)
        
        # Restart camera if it was running
        if was_running:
            self.root.after(1000, self.start_camera)
    
    def open_output_folder(self):
        """Open the output folder in Finder"""
        os.system(f'open "{self.output_dir}"')
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = DocumentScannerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
