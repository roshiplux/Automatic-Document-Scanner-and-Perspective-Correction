"""
Perspective Correction Module
Handles 4-point perspective transformation and image enhancement.
"""

import cv2
import numpy as np


class PerspectiveCorrector:
    """Performs perspective correction on detected documents."""
    
    def __init__(self, output_width=850, output_height=1100):
        """
        Initialize the perspective corrector.
        
        Args:
            output_width: Width of output image in pixels (default: 850 for A4 aspect ratio)
            output_height: Height of output image in pixels (default: 1100 for A4 aspect ratio)
        """
        self.output_width = output_width
        self.output_height = output_height
    
    def calculate_dimensions(self, corners):
        """
        Calculate appropriate output dimensions based on corner points.
        
        Args:
            corners: Ordered array of 4 corner points [TL, TR, BR, BL]
            
        Returns:
            Tuple of (width, height) for output image
        """
        # Unpack points
        tl, tr, br, bl = corners
        
        # Calculate width (max of top and bottom widths)
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))
        
        # Calculate height (max of left and right heights)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))
        
        return max_width, max_height
    
    def apply_perspective_transform(self, image, corners, auto_size=True):
        """
        Apply perspective transformation to correct document orientation.
        
        Args:
            image: Input image
            corners: Ordered array of 4 corner points [TL, TR, BR, BL]
            auto_size: If True, calculate output size from corners; else use preset dimensions
            
        Returns:
            Perspective-corrected image
        """
        # Determine output dimensions
        if auto_size:
            width, height = self.calculate_dimensions(corners)
        else:
            width, height = self.output_width, self.output_height
        
        # Define destination points (rectangle)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply transformation
        warped = cv2.warpPerspective(image, matrix, (width, height))
        
        return warped
    
    def enhance_document(self, image, method='adaptive'):
        """
        Enhance the scanned document for better readability.
        
        Args:
            image: Input image (already perspective-corrected)
            method: Enhancement method ('adaptive', 'otsu', 'color', or 'none')
            
        Returns:
            Enhanced image
        """
        if method == 'none':
            return image
        
        # Convert to grayscale for threshold-based methods
        if method in ['adaptive', 'otsu']:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            if method == 'adaptive':
                # Adaptive thresholding for varied lighting
                enhanced = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11, 10
                )
            else:  # otsu
                # Otsu's thresholding
                _, enhanced = cv2.threshold(
                    gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
            
            return enhanced
        
        elif method == 'color':
            # Enhance color document
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        
        return image
    
    def correct(self, image, corners, auto_size=True, enhance_method='adaptive'):
        """
        Main correction method: applies perspective transform and enhancement.
        
        Args:
            image: Input image
            corners: Ordered array of 4 corner points
            auto_size: If True, calculate output size from corners
            enhance_method: Enhancement method to apply
            
        Returns:
            Corrected and enhanced document image
        """
        # Apply perspective transformation
        warped = self.apply_perspective_transform(image, corners, auto_size)
        
        # Enhance the document
        enhanced = self.enhance_document(warped, enhance_method)
        
        return enhanced
