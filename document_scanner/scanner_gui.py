#!/usr/bin/env python3
"""
Document Scanner - Desktop GUI
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

# Modern Color Scheme
COLORS = {
    'primary': '#00A8CC',        # Bright teal - main actions
    'primary_dark': '#0087A8',   # Darker teal - hover state
    'secondary': '#005082',      # Deep blue - header/sidebar
    'accent': '#FF6B35',         # Warm orange - important actions
    'success': '#06A77D',        # Green - confirmations
    'warning': '#FFB703',        # Amber - warnings
    'danger': '#D62828',         # Red - errors
    'bg_main': '#F5F7FA',        # Very light gray - main background
    'bg_surface': '#FFFFFF',     # White - cards/panels
    'text_primary': '#2C3E50',   # Dark gray - main text
    'text_secondary': '#7F8C8D', # Medium gray - secondary text
    'border': '#E8EEF5',         # Light gray - borders
    'divider': '#DCE1E6',        # Divider lines
}


class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Scanner")
        self.root.geometry("1500x950")
        self.root.configure(bg=COLORS['bg_main'])
        
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
        # Header
        header_frame = tk.Frame(self.root, bg=COLORS['secondary'], height=90)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Logo and title
        header_inner = tk.Frame(header_frame, bg=COLORS['secondary'])
        header_inner.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        title_label = tk.Label(
            header_inner, 
            text="Document Scanner", 
            font=("Segoe UI", 28, "bold"),
            bg=COLORS['secondary'],
            fg=COLORS['bg_surface']
        )
        title_label.pack(side=tk.LEFT, padx=(0, 15))
        
        subtitle_label = tk.Label(
            header_inner,
            text="Professional Document Scanning & Perspective Correction",
            font=("Segoe UI", 11),
            bg=COLORS['secondary'],
            fg=COLORS['border']
        )
        subtitle_label.pack(side=tk.LEFT)
        
        # Main container
        main_container = tk.Frame(self.root, bg=COLORS['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg=COLORS['bg_surface'], width=320, relief=tk.FLAT, borderwidth=0)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 25))
        left_panel.pack_propagate(False)
        
        # Add subtle shadow/border effect
        self._add_shadow(left_panel)
        
        self.setup_controls(left_panel)
        
        # Right panel - Image display
        right_panel = tk.Frame(main_container, bg=COLORS['bg_surface'], relief=tk.FLAT, borderwidth=0)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add subtle shadow/border effect
        self._add_shadow(right_panel)
        
        self.setup_image_display(right_panel)
    
    def _add_shadow(self, frame):
        """Add subtle shadow effect to frame"""
        frame.config(highlightbackground=COLORS['border'], highlightthickness=1)

    def _create_dark_button(self, parent, text, command, font=("Segoe UI", 11, "bold"), pady=10):
        """Create a custom dark button that keeps styling on macOS."""
        btn = tk.Label(
            parent,
            text=text,
            font=font,
            bg="#000000",
            fg="#FFFFFF",
            padx=10,
            pady=pady,
            cursor="hand2",
            relief=tk.SOLID,
            bd=1,
            highlightthickness=0
        )
        btn._enabled = True
        btn._command = command

        def on_enter(_event):
            if btn._enabled:
                btn.config(bg="#1A1A1A")

        def on_leave(_event):
            btn.config(bg="#000000")

        def on_click(_event):
            if btn._enabled and btn._command:
                btn._command()

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<Button-1>", on_click)
        return btn

    def _set_dark_button_state(self, btn, enabled):
        """Enable/disable custom dark button without changing its colors."""
        btn._enabled = enabled
        btn.config(cursor="hand2" if enabled else "arrow")
    
    def setup_controls(self, parent):
        """Setup control buttons and settings"""
        parent.config(bg=COLORS['bg_surface'])
        
        # Action buttons section
        btn_section = tk.Frame(parent, bg=COLORS['bg_surface'])
        btn_section.pack(fill=tk.X, padx=20, pady=(20, 15))
        
        # Section title
        btn_title = tk.Label(
            btn_section,
            text="Actions",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_primary']
        )
        btn_title.pack(anchor=tk.W, pady=(0, 12))
        
        # Upload button
        upload_btn = self._create_dark_button(
            btn_section,
            "Upload Image",
            self.upload_image
        )
        upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Camera button
        self.camera_btn = self._create_dark_button(
            btn_section,
            "Start Camera",
            self.toggle_camera
        )
        self.camera_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Scan button
        self.scan_btn = self._create_dark_button(
            btn_section,
            "Scan Document",
            self.scan_document
        )
        self._set_dark_button_state(self.scan_btn, enabled=False)
        self.scan_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Divider
        divider1 = tk.Frame(parent, bg=COLORS['border'], height=1)
        divider1.pack(fill=tk.X, padx=20, pady=(5, 15))
        
        # Enhancement Mode section
        enhance_section = tk.Frame(parent, bg=COLORS['bg_surface'])
        enhance_section.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        enhance_title = tk.Label(
            enhance_section,
            text="Enhancement Mode",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_primary']
        )
        enhance_title.pack(anchor=tk.W, pady=(0, 10))
        
        modes = [
            ("Original", "none"),
            ("Color Enhanced", "color"),
            ("Adaptive (Best for text)", "adaptive"),
            ("Otsu (High contrast)", "otsu")
        ]
        
        for text, value in modes:
            rb = tk.Radiobutton(
                enhance_section,
                text=text,
                variable=self.enhancement_mode,
                value=value,
                font=("Segoe UI", 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_primary'],
                selectcolor=COLORS['bg_surface'],
                activebackground=COLORS['bg_surface'],
                activeforeground=COLORS['primary'],
                cursor='hand2',
                bd=0,
                highlightthickness=0
            )
            rb.pack(anchor=tk.W, pady=3)
        
        # Divider
        divider2 = tk.Frame(parent, bg=COLORS['border'], height=1)
        divider2.pack(fill=tk.X, padx=20, pady=(5, 15))
        
        # Rotation control section
        rotation_section = tk.Frame(parent, bg=COLORS['bg_surface'])
        rotation_section.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        rotation_title = tk.Label(
            rotation_section,
            text="Camera Rotation",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_primary']
        )
        rotation_title.pack(anchor=tk.W, pady=(0, 10))
        
        auto_rotate_cb = tk.Checkbutton(
            rotation_section,
            text="Auto-rotate (recommended)",
            variable=self.auto_rotate,
            font=("Segoe UI", 10),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_primary'],
            selectcolor=COLORS['primary'],
            activebackground=COLORS['bg_surface'],
            activeforeground=COLORS['primary'],
            cursor='hand2',
            bd=0,
            highlightthickness=0
        )
        auto_rotate_cb.pack(anchor=tk.W, pady=(0, 10))
        
        # Manual rotation buttons
        rotation_btn_frame = tk.Frame(rotation_section, bg=COLORS['bg_surface'])
        rotation_btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        rotation_buttons = [
            ("-90 deg", -90),
            ("+90 deg", 90),
            ("180 deg", 180)
        ]
        
        for btn_text, angle in rotation_buttons:
            btn = self._create_dark_button(
                rotation_btn_frame,
                btn_text,
                lambda a=angle: self.rotate_manual(a),
                font=("Segoe UI", 9, "bold"),
                pady=6
            )
            btn.pack(side=tk.LEFT, padx=2)
        
        # Divider
        divider3 = tk.Frame(parent, bg=COLORS['border'], height=1)
        divider3.pack(fill=tk.X, padx=20, pady=(5, 15))
        
        # Instructions section
        inst_section = tk.Frame(parent, bg=COLORS['bg_surface'])
        inst_section.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        inst_title = tk.Label(
            inst_section,
            text="Setup Guide",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_primary']
        )
        inst_title.pack(anchor=tk.W, pady=(0, 10))
        
        instructions = """Connect Your Mobile Camera:

iPhone Users:
  • Continuity Camera (iOS 16+)
  • Or: EpocCam / iVCam app

Android Users:
  • DroidCam or IP Webcam

Quick Steps:
  1. Connect mobile camera
  2. Place document on dark surface
  3. Click Start Camera
  4. Wait for detection box
  5. Click Scan Document

Pro Tips:
  • Adaptive: handwritten notes
  • Otsu: printed documents
  • Dark background improves detection"""
        
        inst_label = tk.Label(
            inst_section,
            text=instructions,
            font=("Segoe UI", 9),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_secondary'],
            justify=tk.LEFT,
            wraplength=280
        )
        inst_label.pack(anchor=tk.NW)
        
        # Open folder button - at bottom
        folder_btn = self._create_dark_button(
            parent,
            "Open Output Folder",
            self.open_output_folder,
            font=("Segoe UI", 10, "bold"),
            pady=8
        )
        folder_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(0, 20))
    
    def setup_image_display(self, parent):
        """Setup image display area"""
        parent.config(bg=COLORS['bg_surface'])
        
        # Header with status
        header = tk.Frame(parent, bg=COLORS['bg_surface'])
        header.pack(fill=tk.X, padx=20, pady=(20, 15))
        
        # Status label
        self.status_label = tk.Label(
            header,
            text="No image loaded",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_secondary']
        )
        self.status_label.pack(anchor=tk.W)
        
        # Divider
        divider = tk.Frame(parent, bg=COLORS['border'], height=1)
        divider.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        # Image canvas with background
        canvas_container = tk.Frame(parent, bg=COLORS['bg_main'])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.canvas = tk.Label(
            canvas_container,
            bg=COLORS['bg_main'],
            fg=COLORS['text_secondary'],
            font=("Segoe UI", 11)
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
    
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
                self.status_label.config(
                    text="Image loaded - Detecting document...",
                    fg=COLORS['success']
                )
                self.root.update()
                self.process_image()
            else:
                messagebox.showerror("Error", "Failed to load image")
                self.status_label.config(
                    text="Failed to load image",
                    fg=COLORS['danger']
                )
    
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
                        text=f"Camera {cam_idx} connected - Position your document",
                        fg=COLORS['success']
                    )
                    break
                self.cap.release()
        
        if not camera_found:
            messagebox.showinfo(
                "Connect Mobile Camera",
                "Your built-in laptop camera is disabled.\n\n"
                "Connect your mobile phone camera:\n\n"
                "IPHONE USERS\n"
                "1. Install 'Continuity Camera' (iOS 16+)\n"
                "2. OR use 'EpocCam' or 'iVCam' app\n"
                "3. Connect via USB or WiFi\n\n"
                "ANDROID USERS\n"
                "1. Install 'DroidCam' or 'IP Webcam'\n"
                "2. Download companion app on Mac\n"
                "3. Connect via USB or WiFi\n\n"
                "Then click 'Start Camera' again."
            )
            self.status_label.config(
                text="No mobile camera detected",
                fg=COLORS['warning']
            )
            return
        
        self.camera_running = True
        self.camera_btn.config(text="Stop Camera")
        
        self.update_camera()
    
    def stop_camera(self):
        """Stop camera feed"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_btn.config(text="Start Camera")
        self.status_label.config(
            text="Camera stopped",
            fg=COLORS['text_secondary']
        )
    
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
                text="Document detected - Click 'Scan Document' to process",
                fg=COLORS['success']
            )
            self._set_dark_button_state(self.scan_btn, enabled=True)
        else:
            self.status_label.config(
                text="No document detected - Adjust position or lighting",
                fg=COLORS['warning']
            )
            self._set_dark_button_state(self.scan_btn, enabled=False)
        
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
        
        self.status_label.config(text="Scanning and enhancing...", fg=COLORS['primary'])
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
            f"Document scanned successfully!\n\n"
            f"Enhancement: {mode_names[mode]}\n\n"
            f"Saved to output folder"
        )
        
        # Display result
        self.status_label.config(
            text=f"Scan complete - {mode_names[mode]}",
            fg=COLORS['success']
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
