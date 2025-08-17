"""
Image Processing Module for Multimedia Requirements Engineering

This module handles image analysis including OCR, UI component detection,
annotation extraction, and spatial mapping for requirements elicitation.

Author: Cornelius Chimuanya Okechukwu
Institution: Tomas Bata University, Czech Republic
"""

import cv2
import numpy as np
import pytesseract
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import re
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available. Some image processing features may be limited.")

try:
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch/YOLO not available. UI detection features may be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Main class for processing images and extracting requirements-related information."""
    
    def __init__(self, tesseract_config: str = '--oem 3 --psm 6'):
        """
        Initialize the ImageProcessor.
        
        Args:
            tesseract_config: Configuration string for Tesseract OCR
        """
        self.tesseract_config = tesseract_config
        self.ui_model = None
        
        # Load UI detection model if available
        if TORCH_AVAILABLE:
            try:
                self.ui_model = YOLO('yolov8n.pt')  # You can train custom UI model
                logger.info("UI detection model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load UI detection model: {e}")
        
        # Define UI component classes
        self.ui_components = {
            'button': ['button', 'btn', 'submit', 'click', 'press'],
            'input': ['input', 'textbox', 'field', 'entry', 'form'],
            'menu': ['menu', 'navigation', 'nav', 'dropdown'],
            'icon': ['icon', 'symbol', 'image', 'logo'],
            'text': ['text', 'label', 'caption', 'title', 'heading'],
            'container': ['div', 'section', 'panel', 'box', 'frame']
        }

def process_images(image_files: List[str]) -> List[Dict[str, Any]]:
    """
    Process multiple images and extract requirements-related information.
    
    Args:
        image_files: List of image file paths
        
    Returns:
        List of dictionaries containing extracted information for each image
    """
    processor = ImageProcessor()
    requirements = []
    
    for image_file in image_files:
        try:
            logger.info(f"Processing image: {image_file}")
            
            # Load image
            image = cv2.imread(image_file)
            if image is None:
                logger.error(f"Could not load image: {image_file}")
                continue
            
            # Extract text using OCR
            text = extract_text_from_image(image)
            
            # Detect UI components
            ui_elements = detect_ui_components(image)
            
            # Extract annotations (colored regions, arrows, etc.)
            annotations = extract_annotations(image)
            
            # Analyze visual elements
            visual_analysis = analyze_visual_elements(image)
            
            # Map spatial annotations
            spatial_mapping = map_spatial_annotations(image, annotations)
            
            # Generate requirements from image analysis
            image_requirements = generate_requirements_from_image(
                text, ui_elements, annotations, visual_analysis, spatial_mapping
            )
            
            image_result = {
                'image_file': image_file,
                'extracted_text': text,
                'ui_elements': ui_elements,
                'annotations': annotations,
                'visual_analysis': visual_analysis,
                'spatial_mapping': spatial_mapping,
                'requirements': image_requirements,
                'timestamp': None,  # To be filled by synchronization
                'quality_score': assess_image_quality(image)
            }
            
            requirements.append(image_result)
            
        except Exception as e:
            logger.error(f"Error processing image {image_file}: {e}")
            continue
    
    return requirements

def extract_text_from_image(image: np.ndarray, 
                          languages: str = 'eng+ces') -> Dict[str, Any]:
    """
    Extract text from image using OCR.
    
    Args:
        image: Input image as numpy array
        languages: Languages for OCR (English + Czech)
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    try:
        # Preprocess image for better OCR
        preprocessed = preprocess_for_ocr(image)
        
        # Extract text with confidence scores
        data = pytesseract.image_to_data(
            preprocessed, 
            lang=languages,
            config='--oem 3 --psm 6',
            output_type=pytesseract.Output.DICT
        )
        
        # Filter out low-confidence text
        min_confidence = 30
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > min_confidence:
                word_info = {
                    'text': data['text'][i].strip(),
                    'confidence': int(data['conf'][i]),
                    'bbox': (data['left'][i], data['top'][i], 
                            data['width'][i], data['height'][i])
                }
                if word_info['text']:  # Skip empty strings
                    words.append(word_info)
        
        # Combine words into sentences
        full_text = ' '.join([word['text'] for word in words])
        
        # Extract UI-related keywords
        ui_keywords = extract_ui_keywords(full_text)
        
        return {
            'full_text': full_text,
            'words': words,
            'ui_keywords': ui_keywords,
            'word_count': len(words),
            'avg_confidence': np.mean([w['confidence'] for w in words]) if words else 0
        }
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return {
            'full_text': '',
            'words': [],
            'ui_keywords': [],
            'word_count': 0,
            'avg_confidence': 0
        }

def detect_ui_components(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect UI components in the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected UI components with bounding boxes
    """
    ui_elements = []
    
    try:
        # Method 1: Template matching for common UI elements
        template_matches = detect_ui_templates(image)
        ui_elements.extend(template_matches)
        
        # Method 2: Contour-based detection
        contour_elements = detect_ui_contours(image)
        ui_elements.extend(contour_elements)
        
        # Method 3: YOLO-based detection (if available)
        if TORCH_AVAILABLE:
            yolo_elements = detect_ui_yolo(image)
            ui_elements.extend(yolo_elements)
        
        # Remove duplicates and merge overlapping detections
        ui_elements = merge_overlapping_detections(ui_elements)
        
        return ui_elements
        
    except Exception as e:
        logger.error(f"UI component detection failed: {e}")
        return []

def extract_annotations(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extract user annotations from the image (arrows, highlights, drawings).
    
    Args:
        image: Input image as numpy array
        
    Returns:
        List of detected annotations
    """
    annotations = []
    
    try:
        # Detect colored regions (highlights)
        colored_regions = detect_colored_annotations(image)
        annotations.extend(colored_regions)
        
        # Detect arrows and lines
        arrows_lines = detect_arrows_and_lines(image)
        annotations.extend(arrows_lines)
        
        # Detect text annotations (added text)
        text_annotations = detect_text_annotations(image)
        annotations.extend(text_annotations)
        
        # Detect geometric shapes (circles, rectangles)
        geometric_shapes = detect_geometric_annotations(image)
        annotations.extend(geometric_shapes)
        
        return annotations
        
    except Exception as e:
        logger.error(f"Annotation extraction failed: {e}")
        return []

def analyze_visual_elements(image: np.ndarray) -> Dict[str, Any]:
    """
    Analyze visual characteristics of the image.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary containing visual analysis results
    """
    try:
        height, width = image.shape[:2]
        
        # Color analysis
        color_analysis = analyze_color_distribution(image)
        
        # Layout analysis
        layout_analysis = analyze_layout_structure(image)
        
        # Visual complexity
        complexity_score = calculate_visual_complexity(image)
        
        # Focus regions (where user attention might be drawn)
        focus_regions = detect_focus_regions(image)
        
        return {
            'dimensions': {'width': width, 'height': height},
            'color_analysis': color_analysis,
            'layout_analysis': layout_analysis,
            'complexity_score': complexity_score,
            'focus_regions': focus_regions,
            'aspect_ratio': width / height if height > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Visual analysis failed: {e}")
        return {}

def map_spatial_annotations(image: np.ndarray, 
                          annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create spatial mapping of annotations to UI regions.
    
    Args:
        image: Input image as numpy array
        annotations: List of detected annotations
        
    Returns:
        Dictionary containing spatial mapping information
    """
    try:
        height, width = image.shape[:2]
        
        # Create spatial grid (divide image into regions)
        grid_size = 10
        spatial_grid = create_spatial_grid(width, height, grid_size)
        
        # Map annotations to grid regions
        annotation_mapping = {}
        for i, annotation in enumerate(annotations):
            if 'bbox' in annotation:
                x, y, w, h = annotation['bbox']
                grid_coords = get_grid_coordinates(x, y, w, h, width, height, grid_size)
                annotation_mapping[f"annotation_{i}"] = {
                    'annotation': annotation,
                    'grid_coords': grid_coords,
                    'relative_position': get_relative_position(x, y, w, h, width, height)
                }
        
        # Identify annotation clusters
        clusters = find_annotation_clusters(annotations)
        
        return {
            'spatial_grid': spatial_grid,
            'annotation_mapping': annotation_mapping,
            'clusters': clusters,
            'total_annotations': len(annotations)
        }
        
    except Exception as e:
        logger.error(f"Spatial mapping failed: {e}")
        return {}

def generate_requirements_from_image(text: Dict[str, Any],
                                   ui_elements: List[Dict[str, Any]],
                                   annotations: List[Dict[str, Any]],
                                   visual_analysis: Dict[str, Any],
                                   spatial_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate requirements based on image analysis.
    
    Args:
        text: Extracted text information
        ui_elements: Detected UI components
        annotations: User annotations
        visual_analysis: Visual characteristic analysis
        spatial_mapping: Spatial annotation mapping
        
    Returns:
        List of extracted requirements
    """
    requirements = []
    
    try:
        # Extract requirements from text annotations
        text_requirements = extract_text_based_requirements(text)
        requirements.extend(text_requirements)
        
        # Extract requirements from UI element analysis
        ui_requirements = extract_ui_based_requirements(ui_elements)
        requirements.extend(ui_requirements)
        
        # Extract requirements from user annotations
        annotation_requirements = extract_annotation_based_requirements(annotations)
        requirements.extend(annotation_requirements)
        
        # Extract requirements from spatial relationships
        spatial_requirements = extract_spatial_requirements(spatial_mapping)
        requirements.extend(spatial_requirements)
        
        # Extract requirements from visual design analysis
        visual_requirements = extract_visual_design_requirements(visual_analysis)
        requirements.extend(visual_requirements)
        
        # Add metadata to requirements
        for req in requirements:
            req['source'] = 'image_analysis'
            req['confidence'] = calculate_requirement_confidence(req)
            req['priority'] = determine_requirement_priority(req, annotations)
        
        return requirements
        
    except Exception as e:
        logger.error(f"Requirements generation failed: {e}")
        return []

# Helper functions

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.medianBlur(gray, 3)
    
    # Contrast enhancement
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(denoised)
    
    # Binarization
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def extract_ui_keywords(text: str) -> List[str]:
    """Extract UI-related keywords from text."""
    ui_keywords = []
    text_lower = text.lower()
    
    # Common UI terms
    ui_terms = [
        'button', 'menu', 'icon', 'field', 'form', 'input', 'dropdown',
        'checkbox', 'radio', 'slider', 'tab', 'panel', 'dialog', 'popup',
        'navigation', 'search', 'filter', 'sort', 'save', 'cancel', 'submit',
        'login', 'logout', 'register', 'profile', 'settings', 'help'
    ]
    
    for term in ui_terms:
        if term in text_lower:
            ui_keywords.append(term)
    
    return list(set(ui_keywords))

def detect_ui_templates(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect UI elements using template matching."""
    # This would use pre-defined templates for common UI elements
    # For now, return empty list - implement based on specific UI patterns
    return []

def detect_ui_contours(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect UI elements using contour analysis."""
    ui_elements = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size and aspect ratio to identify potential UI elements
            if 100 < area < 10000:  # Reasonable size for UI elements
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio
                    ui_elements.append({
                        'type': 'contour_element',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'confidence': min(area / 1000, 1.0)  # Simple confidence metric
                    })
    
    except Exception as e:
        logger.error(f"Contour detection failed: {e}")
    
    return ui_elements

def detect_ui_yolo(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect UI elements using YOLO model (if available)."""
    # Placeholder for YOLO-based UI detection
    # Would require custom-trained model for UI elements
    return []

def merge_overlapping_detections(detections: List[Dict[str, Any]], 
                               overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Merge overlapping UI element detections."""
    if not detections:
        return []
    
    # Simple non-maximum suppression
    merged = []
    used = [False] * len(detections)
    
    for i, det1 in enumerate(detections):
        if used[i]:
            continue
            
        current_group = [det1]
        used[i] = True
        
        for j, det2 in enumerate(detections[i+1:], i+1):
            if used[j]:
                continue
                
            if calculate_overlap(det1.get('bbox'), det2.get('bbox')) > overlap_threshold:
                current_group.append(det2)
                used[j] = True
        
        # Merge group into single detection
        merged_detection = merge_detection_group(current_group)
        merged.append(merged_detection)
    
    return merged

def calculate_overlap(bbox1: Optional[Tuple], bbox2: Optional[Tuple]) -> float:
    """Calculate overlap between two bounding boxes."""
    if not bbox1 or not bbox2:
        return 0.0
    
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def merge_detection_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a group of overlapping detections."""
    if len(group) == 1:
        return group[0]
    
    # Calculate bounding box that encompasses all detections
    bboxes = [det.get('bbox') for det in group if det.get('bbox')]
    if not bboxes:
        return group[0]
    
    x_min = min(bbox[0] for bbox in bboxes)
    y_min = min(bbox[1] for bbox in bboxes)
    x_max = max(bbox[0] + bbox[2] for bbox in bboxes)
    y_max = max(bbox[1] + bbox[3] for bbox in bboxes)
    
    merged_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    # Average confidence
    confidences = [det.get('confidence', 0) for det in group]
    avg_confidence = sum(confidences) / len(confidences)
    
    return {
        'type': 'merged_element',
        'bbox': merged_bbox,
        'confidence': avg_confidence,
        'merged_count': len(group),
        'original_types': [det.get('type') for det in group]
    }

def detect_colored_annotations(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect colored annotations (highlights, markers)."""
    annotations = []
    
    try:
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for common annotation colors
        color_ranges = {
            'red': [(0, 50, 50), (10, 255, 255)],
            'yellow': [(20, 50, 50), (30, 255, 255)],
            'green': [(40, 50, 50), (80, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)]
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum annotation size
                    x, y, w, h = cv2.boundingRect(contour)
                    annotations.append({
                        'type': 'colored_annotation',
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(area / 1000, 1.0)
                    })
    
    except Exception as e:
        logger.error(f"Colored annotation detection failed: {e}")
    
    return annotations

def detect_arrows_and_lines(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect arrows and lines in annotations."""
    annotations = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Simple arrow detection (lines with significant length)
                if length > 50:
                    annotations.append({
                        'type': 'line_annotation',
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': np.arctan2(y2-y1, x2-x1) * 180 / np.pi,
                        'confidence': min(length / 200, 1.0)
                    })
    
    except Exception as e:
        logger.error(f"Arrow/line detection failed: {e}")
    
    return annotations

def detect_text_annotations(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect added text annotations."""
    # This would identify text that appears to be added annotations
    # vs. original UI text - complex implementation needed
    return []

def detect_geometric_annotations(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect geometric shape annotations (circles, rectangles)."""
    annotations = []
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect circles
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                annotations.append({
                    'type': 'circle_annotation',
                    'center': (x, y),
                    'radius': r,
                    'bbox': (x-r, y-r, 2*r, 2*r),
                    'confidence': 0.7
                })
    
    except Exception as e:
        logger.error(f"Geometric annotation detection failed: {e}")
    
    return annotations

def analyze_color_distribution(image: np.ndarray) -> Dict[str, Any]:
    """Analyze color distribution in the image."""
    try:
        # Calculate color histograms
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Calculate dominant colors
        dominant_colors = extract_dominant_colors(image, k=5)
        
        return {
            'histograms': {
                'blue': hist_b.flatten().tolist(),
                'green': hist_g.flatten().tolist(),
                'red': hist_r.flatten().tolist()
            },
            'dominant_colors': dominant_colors
        }
    except Exception as e:
        logger.error(f"Color analysis failed: {e}")
        return {}

def extract_dominant_colors(image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
    """Extract dominant colors using k-means clustering."""
    try:
        from sklearn.cluster import KMeans
        
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and return
        centers = np.uint8(centers)
        return [tuple(color) for color in centers]
        
    except Exception as e:
        logger.error(f"Dominant color extraction failed: {e}")
        return []

def analyze_layout_structure(image: np.ndarray) -> Dict[str, Any]:
    """Analyze the layout structure of the interface."""
    # This would analyze grid patterns, alignment, spacing, etc.
    # Simplified implementation
    height, width = image.shape[:2]
    
    return {
        'grid_detected': False,  # Would implement grid detection
        'alignment_score': 0.5,  # Would calculate alignment metrics
        'spacing_consistency': 0.5,  # Would analyze spacing patterns
        'layout_type': 'unknown'  # Would classify layout type
    }

def calculate_visual_complexity(image: np.ndarray) -> float:
    """Calculate visual complexity score of the image."""
    try:
        # Use edge density as a proxy for visual complexity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Normalize to 0-1 scale
        complexity_score = min(edge_density * 10, 1.0)
        
        return complexity_score
        
    except Exception as e:
        logger.error(f"Complexity calculation failed: {e}")
        return 0.0

def detect_focus_regions(image: np.ndarray) -> List[Dict[str, Any]]:
    """Detect regions where user attention might be focused."""
    # This would use saliency detection or other attention models
    # Simplified implementation
    return []

def create_spatial_grid(width: int, height: int, grid_size: int) -> Dict[str, Any]:
    """Create spatial grid for annotation mapping."""
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    return {
        'grid_size': grid_size,
        'cell_width': cell_width,
        'cell_height': cell_height,
        'total_cells': grid_size * grid_size
    }

def get_grid_coordinates(x: int, y: int, w: int, h: int, 
                        width: int, height: int, grid_size: int) -> Tuple[int, int]:
    """Get grid coordinates for a bounding box."""
    cell_width = width // grid_size
    cell_height = height // grid_size
    
    center_x = x + w // 2
    center_y = y + h // 2
    
    grid_x = min(center_x // cell_width, grid_size - 1)
    grid_y = min(center_y // cell_height, grid_size - 1)
    
    return (int(grid_x), int(grid_y))

def get_relative_position(x: int, y: int, w: int, h: int, 
                         width: int, height: int) -> Dict[str, float]:
    """Get relative position of element in image."""
    center_x = (x + w / 2) / width
    center_y = (y + h / 2) / height
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'relative_size': (w * h) / (width * height)
    }

def find_annotation_clusters(annotations: List[Dict[str, Any]]) -> List[List[int]]:
    """Find clusters of nearby annotations."""
    # Simple clustering based on proximity
    clusters = []
    used = [False] * len(annotations)
    
    for i, ann1 in enumerate(annotations):
        if used[i]:
            continue
            
        cluster = [i]
        used[i] = True
        
        for j, ann2 in enumerate(annotations[i+1:], i+1):
            if used[j]:
                continue
                
            # Check if annotations are close enough to cluster
            if are_annotations_close(ann1, ann2):
                cluster.append(j)
                used[j] = True
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters

def are_annotations_close(ann1: Dict[str, Any], ann2: Dict[str, Any], 
                         distance_threshold: float = 100.0) -> bool:
    """Check if two annotations are close enough to be clustered."""
    # Get center points
    bbox1 = ann1.get('bbox')
    bbox2 = ann2.get('bbox')
    
    if not bbox1 or not bbox2:
        return False
    
    center1 = (bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2)
    center2 = (bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2)
    
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    return distance < distance_threshold

def extract_text_based_requirements(text: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from text annotations."""
    requirements = []
    
    full_text = text.get('full_text', '')
    ui_keywords = text.get('ui_keywords', [])
    
    # Look for requirement-indicating phrases
    requirement_patterns = [
        r'(need|want|should|must|require).{1,100}',
        r'(add|include|provide|implement).{1,100}',
        r'(improve|enhance|better|fix).{1,100}',
        r'(here|there)\s+(should|could|need).{1,50}'
    ]
    
    for pattern in requirement_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            requirement_text = match.group().strip()
            requirements.append({
                'id': f"IMG-TXT-{len(requirements)+1}",
                'type': 'functional',
                'description': requirement_text,
                'source_text': full_text,
                'ui_keywords': ui_keywords,
                'extraction_method': 'text_pattern'
            })
    
    return requirements

def extract_ui_based_requirements(ui_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from UI element analysis."""
    requirements = []
    
    # Analyze UI element density and distribution
    if len(ui_elements) > 10:
        requirements.append({
            'id': f"IMG-UI-1",
            'type': 'usability',
            'description': 'Interface appears cluttered with many UI elements. Consider simplification.',
            'ui_element_count': len(ui_elements),
            'extraction_method': 'ui_density_analysis'
        })
    
    # Analyze UI element types
    element_types = [elem.get('type', 'unknown') for elem in ui_elements]
    type_counts = {t: element_types.count(t) for t in set(element_types)}
    
    for elem_type, count in type_counts.items():
        if count > 5:
            requirements.append({
                'id': f"IMG-UI-{elem_type}",
                'type': 'usability',
                'description': f'High number of {elem_type} elements detected. Review for consistency.',
                'element_type': elem_type,
                'count': count,
                'extraction_method': 'ui_type_analysis'
            })
    
    return requirements

def extract_annotation_based_requirements(annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract requirements from user annotations."""
    requirements = []
    
    for i, annotation in enumerate(annotations):
        ann_type = annotation.get('type', 'unknown')
        
        if ann_type == 'colored_annotation':
            color = annotation.get('color', 'unknown')
            requirements.append({
                'id': f"IMG-ANN-{i+1}",
                'type': 'usability',
                'description': f'User highlighted area with {color} - likely indicates area of interest or needed improvement',
                'annotation_type': ann_type,
                'color': color,
                'bbox': annotation.get('bbox'),
                'extraction_method': 'annotation_analysis'
            })
        
        elif ann_type == 'line_annotation':
            requirements.append({
                'id': f"IMG-ANN-{i+1}",
                'type': 'usability', 
                'description': 'User drew line/arrow - likely indicating desired connection or flow',
                'annotation_type': ann_type,
                'length': annotation.get('length'),
                'extraction_method': 'annotation_analysis'
            })
    
    return requirements

def extract_spatial_requirements(spatial_mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from spatial relationships."""
    requirements = []
    
    clusters = spatial_mapping.get('clusters', [])
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 2:
            requirements.append({
                'id': f"IMG-SPATIAL-{i+1}",
                'type': 'usability',
                'description': f'Multiple annotations clustered in same area - likely indicates problem region requiring attention',
                'cluster_size': len(cluster),
                'extraction_method': 'spatial_analysis'
            })
    
    return requirements

def extract_visual_design_requirements(visual_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract requirements from visual design analysis."""
    requirements = []
    
    complexity_score = visual_analysis.get('complexity_score', 0)
    
    if complexity_score > 0.7:
        requirements.append({
            'id': 'IMG-VISUAL-1',
            'type': 'usability',
            'description': 'Interface appears visually complex. Consider simplifying design for better user experience.',
            'complexity_score': complexity_score,
            'extraction_method': 'visual_complexity_analysis'
        })
    
    return requirements

def calculate_requirement_confidence(requirement: Dict[str, Any]) -> float:
    """Calculate confidence score for a requirement."""
    # Simple confidence calculation based on extraction method
    method_confidences = {
        'text_pattern': 0.8,
        'annotation_analysis': 0.9,
        'ui_density_analysis': 0.6,
        'ui_type_analysis': 0.7,
        'spatial_analysis': 0.75,
        'visual_complexity_analysis': 0.5
    }
    
    method = requirement.get('extraction_method', 'unknown')
    return method_confidences.get(method, 0.5)

def determine_requirement_priority(requirement: Dict[str, Any], 
                                 annotations: List[Dict[str, Any]]) -> str:
    """Determine priority of requirement based on context."""
    # Higher priority for requirements with multiple supporting annotations
    if requirement.get('extraction_method') == 'annotation_analysis':
        return 'high'
    elif requirement.get('extraction_method') == 'text_pattern':
        return 'medium'
    else:
        return 'low'

def assess_image_quality(image: np.ndarray) -> float:
    """Assess the quality of the image for processing."""
    try:
        # Check image sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize sharpness score (higher is better)
        sharpness_score = min(sharpness / 1000, 1.0)
        
        # Check brightness
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
        
        # Overall quality score
        quality_score = (sharpness_score + brightness_score) / 2
        
        return quality_score
        
    except Exception as e:
        logger.error(f"Image quality assessment failed: {e}")
        return 0.0
