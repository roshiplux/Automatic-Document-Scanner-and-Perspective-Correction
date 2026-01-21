"""
Simple, Fast, and RELIABLE Document Detector
Uses proven OpenCV techniques - optimized for real-time detection
"""

import cv2
import numpy as np


class SimpleDocumentDetector:
    """Fast and reliable document detection using simplified approach."""
    
    def __init__(self):
        """Initialize detector with optimal settings."""
        self.min_area = 0.01  # Minimum 1% of image
        self.max_area = 0.95  # Maximum 95% of image
        
    def detect(self, image):
        """
        Detect document in image using fast, reliable method.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (success, corners, visualization)
        """
        # Create visualization
        vis = image.copy()
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 3. Adaptive threshold - works best for varied lighting
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 4. Invert if needed (document should be white on black)
        if np.mean(thresh) > 127:
            thresh = 255 - thresh
        
        # 5. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, vis
        
        # 6. Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Calculate area constraints
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * self.min_area
        max_area = img_area * self.max_area
        
        # 7. Find best rectangle
        for contour in contours[:10]:  # Check top 10
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # Approximate to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Found rectangle
            if len(approx) == 4:
                # Simple validation: check if convex
                if cv2.isContourConvex(approx):
                    corners = self._order_points(approx.reshape(4, 2))
                    
                    # Draw on visualization
                    pts = corners.reshape(-1, 1, 2).astype(np.int32)
                    cv2.drawContours(vis, [pts], -1, (0, 255, 0), 3)
                    
                    # Draw corners
                    for i, pt in enumerate(corners):
                        cv2.circle(vis, tuple(pt.astype(int)), 10, (0, 0, 255), -1)
                        cv2.putText(vis, str(i+1), tuple(pt.astype(int)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    return True, corners, vis
        
        return False, None, vis
    
    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect
