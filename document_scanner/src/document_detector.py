"""
Document Detection Module
Handles edge detection, contour finding, and corner detection for document scanning.
Power-packed version for challenging images.
"""

import cv2
import numpy as np
from collections import deque


class DocumentDetector:
    """Detects document boundaries in images using aggressive multi-strategy detection."""
    
    def __init__(self, min_area_ratio=0.005, max_area_ratio=0.99):
        """
        Initialize the document detector with ULTRA-AGGRESSIVE settings.
        
        Args:
            min_area_ratio: Minimum document area as ratio of image area (default: 0.005 - ultra small)
            max_area_ratio: Maximum document area as ratio of image area (default: 0.99 - ultra large)
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        
        # Stronger temporal smoothing for ultra-stable detection
        self.recent_detections = deque(maxlen=10)  # Store last 10 detections (was 5)
        self.stable_corners = None
        self.stable_confidence = 0
    
    def preprocess_image(self, image):
        """
        ULTRA-ENHANCED preprocessing with maximum contrast boost.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Stronger bilateral filter
        filtered = cv2.bilateralFilter(gray, 11, 100, 100)
        
        # Aggressive CLAHE for maximum contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        # Additional sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
    
    def detect_edges(self, gray_image, color_image=None):
        """
        Aggressive multi-strategy edge detection for challenging images.
        
        Args:
            gray_image: Grayscale image
            color_image: Original BGR image (optional, for color-based detection)
            
        Returns:
            Edge map (binary image)
        """
        edges_list = []
        
        # Strategy 1: Multiple Canny thresholds
        median = np.median(gray_image)
        
        # Aggressive (catches more edges)
        lower1 = int(max(0, 0.33 * median))
        upper1 = int(min(255, 1.0 * median))
        edges1 = cv2.Canny(gray_image, lower1, upper1)
        edges_list.append(edges1)
        
        # Standard
        lower2 = int(max(0, 0.66 * median))
        upper2 = int(min(255, 1.33 * median))
        edges2 = cv2.Canny(gray_image, lower2, upper2)
        edges_list.append(edges2)
        
        # Conservative (only strong edges)
        edges3 = cv2.Canny(gray_image, 50, 150)
        edges_list.append(edges3)
        
        # Strategy 2: Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(gray_image, cv2.MORPH_GRADIENT, kernel)
        _, edges4 = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges_list.append(edges4)
        
        # Strategy 3: Sobel edges (multiple kernel sizes)
        sobelx3 = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely3 = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag3 = np.sqrt(sobelx3**2 + sobely3**2)
        sobel_mag3 = np.uint8(sobel_mag3 * 255 / (sobel_mag3.max() + 1e-10))
        _, edges5 = cv2.threshold(sobel_mag3, 30, 255, cv2.THRESH_BINARY)
        edges_list.append(edges5)
        
        # Strategy 4: Sobel with larger kernel
        sobelx5 = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
        sobely5 = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
        sobel_mag5 = np.sqrt(sobelx5**2 + sobely5**2)
        sobel_mag5 = np.uint8(sobel_mag5 * 255 / (sobel_mag5.max() + 1e-10))
        _, edges6 = cv2.threshold(sobel_mag5, 30, 255, cv2.THRESH_BINARY)
        edges_list.append(edges6)
        
        # Strategy 5: Adaptive thresholding (multiple block sizes)
        adaptive1 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        edges7 = cv2.Canny(adaptive1, 30, 100)
        edges_list.append(edges7)
        
        adaptive2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 15, 2)
        edges8 = cv2.Canny(adaptive2, 30, 100)
        edges_list.append(edges8)
        
        # Strategy 6: Color-based edge detection (if color image provided)
        if color_image is not None:
            # LAB color space
            lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            edges9 = cv2.Canny(l_channel, 30, 120)
            edges_list.append(edges9)
            
            # HSV color space
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            edges10 = cv2.Canny(v_channel, 30, 120)
            edges_list.append(edges10)
        
        # Strategy 7: Laplacian
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        _, edges11 = cv2.threshold(laplacian, 20, 255, cv2.THRESH_BINARY)
        edges_list.append(edges11)
        
        # Combine all strategies
        edges = np.zeros_like(edges1)
        for e in edges_list:
            edges = cv2.bitwise_or(edges, e)
        
        # Aggressive morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
        
        # Strong dilation to connect edges
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel_dilate, iterations=3)
        
        return edges
    
    def find_document_contour(self, image, edges):
        """
        Aggressive document contour detection with multiple strategies.
        
        Args:
            image: Original image for size reference
            edges: Edge-detected binary image
            
        Returns:
            Document contour (4 points) or None if not found
        """
        # Find contours with different retrieval modes
        contours_list = []
        
        # Method 1: External contours only
        contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.extend(contours1)
        
        # Method 2: All contours
        contours2, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.extend(contours2)
        
        # Remove duplicates and sort by area
        contours = sorted(contours_list, key=cv2.contourArea, reverse=True)
        
        # Calculate area constraints
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio
        
        # ULTRA-AGGRESSIVE: Try top 100 contours with many epsilon values
        for contour in contours[:100]:  # Check top 100 largest contours (was 30)
            area = cv2.contourArea(contour)
            
            # Skip if area is not within acceptable range
            if area < min_area or area > max_area:
                continue
            
            # Get perimeter
            peri = cv2.arcLength(contour, True)
            
            if peri == 0:
                continue
            
            # Try MANY epsilon values for approximation
            for epsilon_multiplier in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]:
                approx = cv2.approxPolyDP(contour, epsilon_multiplier * peri, True)
                
                # If we have 4 points, check if it's valid
                if len(approx) == 4:
                    # Ultra-lenient checks
                    if cv2.isContourConvex(approx):
                        # Very relaxed angle constraint
                        if self._check_angle_constraints_ultra_relaxed(approx):
                            return approx
        
        # LAST RESORT: Accept ANY convex 4-point shape in top 150 contours
        for contour in contours[:150]:
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            peri = cv2.arcLength(contour, True)
            if peri == 0:
                continue
            
            for epsilon_multiplier in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]:
                approx = cv2.approxPolyDP(contour, epsilon_multiplier * peri, True)
                
                if len(approx) == 4:
                    # Accept any convex quadrilateral without angle checks
                    if cv2.isContourConvex(approx):
                        return approx
        
        return None
    
    def _check_angle_constraints_relaxed(self, contour):
        """
        Relaxed angle checking for challenging images.
        
        Args:
            contour: 4-point contour
            
        Returns:
            Boolean indicating if angles are acceptable
        """
        if len(contour) != 4:
            return False
        
        # Calculate angles at each corner
        points = contour.reshape(4, 2)
        
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # More relaxed: 30-150 degrees
            if angle_deg < 30 or angle_deg > 150:
                return False
        
        return True
    
    def _check_angle_constraints_ultra_relaxed(self, contour):
        """
        ULTRA-relaxed angle checking - accepts almost anything.
        
        Args:
            contour: 4-point contour
            
        Returns:
            Boolean indicating if angles are acceptable
        """
        if len(contour) != 4:
            return False
        
        # Calculate angles at each corner
        points = contour.reshape(4, 2)
        
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # ULTRA-relaxed: 20-160 degrees (almost anything)
            if angle_deg < 20 or angle_deg > 160:
                return False
        
        return True
    
    def order_points(self, pts):
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        # Initialize ordered points array
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Top-left point has the smallest sum
        # Bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right has smallest difference
        # Bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def detect(self, image, use_temporal_smoothing=True):
        """
        Enhanced detection with temporal smoothing for stability.
        
        Args:
            image: Input BGR image
            use_temporal_smoothing: Use frame-to-frame smoothing for stability
            
        Returns:
            Tuple of (success, corners, visualization)
            - success: Boolean indicating if document was found
            - corners: Ordered array of 4 corner points or None
            - visualization: Image with detected contour drawn
        """
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Preprocess with enhanced method
        gray = self.preprocess_image(image)
        
        # Detect edges with multi-strategy approach (pass color image too)
        edges = self.detect_edges(gray, image)
        
        # Find document contour
        contour = self.find_document_contour(image, edges)
        
        if contour is not None:
            # Order the points
            corners = self.order_points(contour.reshape(4, 2))
            
            # Apply temporal smoothing for stable detection
            if use_temporal_smoothing:
                corners = self._apply_temporal_smoothing(corners)
                if corners is None:
                    return False, None, vis_image
            
            # Draw on visualization with thicker, more visible lines
            contour_reshaped = corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(vis_image, [contour_reshaped], -1, (0, 255, 0), 4)
            
            # Draw corners with labels
            corner_labels = ['TL', 'TR', 'BR', 'BL']
            for i, (point, label) in enumerate(zip(corners, corner_labels)):
                pt = tuple(point.astype(int))
                cv2.circle(vis_image, pt, 12, (0, 0, 255), -1)
                cv2.circle(vis_image, pt, 14, (255, 255, 255), 2)
                # Add label
                cv2.putText(vis_image, label, (pt[0] - 10, pt[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return True, corners, vis_image
        
        # No detection - reset temporal tracking
        if use_temporal_smoothing:
            self.recent_detections.clear()
            self.stable_corners = None
            self.stable_confidence = 0
        
        return False, None, vis_image
    
    def _apply_temporal_smoothing(self, corners):
        """
        Apply temporal smoothing for stable detection across frames.
        Prevents jittery detection.
        
        Args:
            corners: Newly detected corners
            
        Returns:
            Smoothed corners or None if not stable enough
        """
        # Add to recent detections
        self.recent_detections.append(corners)
        
        # Ultra-fast smoothing - only need 2 detections
        if len(self.recent_detections) < 2:
            return corners
        
        # Strong averaging with more weight on recent detections
        weights = np.linspace(0.5, 1.0, len(self.recent_detections))
        weights = weights / weights.sum()
        
        avg_corners = np.average(list(self.recent_detections), axis=0, weights=weights)
        
        # Calculate stability (how much corners are moving)
        if self.stable_corners is not None:
            movement = np.mean(np.linalg.norm(avg_corners - self.stable_corners, axis=1))
            
            # Very lenient movement threshold
            if movement < 20:  # pixels (was 10)
                self.stable_confidence = min(10, self.stable_confidence + 2)
            else:
                self.stable_confidence = max(0, self.stable_confidence - 0.5)
        
        # Update stable corners
        self.stable_corners = avg_corners
        
        # Return smoothed corners
        return avg_corners.astype(np.float32)
