#!/usr/bin/env python3
"""
Signature Cropper API
FastAPI endpoint that uses the exact signature detection logic from the MOHRE OCR Colab notebook
"""

import os
import sys
import json
import logging
import tempfile
import io
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract
import google.generativeai as genai
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectedZone:
    """Represents a detected signature zone"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    text_above: str

@dataclass
class GeminiClassificationResult:
    """Result from Gemini classification"""
    classification: str  # "signature", "contract", or "blank"
    confidence: float
    reasoning: str
    gemini_response: str

class GeminiZoneClassifier:
    """Enhanced signature zone classifier using Gemini for classification"""

    def __init__(self, api_key: str, verbose: bool = False):
        """Initialize the classifier with Gemini API key"""
        self.verbose = verbose
        self.setup_gemini(api_key)

        # Check if tesseract is available
        try:
            self.tesseract_path = self._find_tesseract()
            if self.tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
                logger.info(f"Tesseract found at: {self.tesseract_path}")
            else:
                logger.warning("Tesseract not found. OCR functionality may be limited.")
        except Exception as e:
            logger.error(f"Error setting up Tesseract: {e}")

    def setup_gemini(self, api_key: str):
        """Setup Gemini API"""
        try:
            genai.configure(api_key=api_key)
            # Use Gemini 2.5 Pro with thinking (most advanced model with superior reasoning)
            self.model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Gemini 2.5 Pro with thinking initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _find_tesseract(self) -> Optional[str]:
        """Find tesseract executable"""
        common_paths = [
            '/opt/homebrew/bin/tesseract',  # macOS Homebrew
            '/usr/local/bin/tesseract',     # macOS/Linux
            '/usr/bin/tesseract',           # Linux
            'tesseract'                     # System PATH
        ]

        for path in common_paths:
            if os.path.exists(path) or (path == 'tesseract' and os.system('which tesseract > /dev/null 2>&1') == 0):
                return path
        return None

    def extract_second_page(self, pdf_path: str) -> np.ndarray:
        """Extract the second page from PDF as image"""
        try:
            pdf_document = fitz.open(pdf_path)

            if len(pdf_document) < 2:
                raise ValueError(f"PDF has only {len(pdf_document)} page(s), need at least 2")

            # Get second page (index 1)
            page = pdf_document[1]

            # Convert to image with high DPI for better OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = 144 DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")

            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            pdf_document.close()

            if self.verbose:
                logger.info(f"Extracted second page: {image.shape}")

            return image

        except Exception as e:
            logger.error(f"Error extracting second page: {e}")
            raise

    def extract_image_from_upload(self, file_content: bytes, filename: str) -> np.ndarray:
        """Extract image from uploaded file (PDF or image)"""
        try:
            # Check if it's a PDF file
            if filename.lower().endswith('.pdf'):
                # Save to temporary file to process with PyMuPDF
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file.flush()
                    
                    # Extract second page from PDF
                    image = self.extract_second_page(tmp_file.name)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                    return image
            else:
                # Handle as image file
                nparr = np.frombuffer(file_content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Could not decode image file")
                    
                if self.verbose:
                    logger.info(f"Extracted image: {image.shape}")
                    
                return image

        except Exception as e:
            logger.error(f"Error extracting image from upload: {e}")
            raise

    def get_bottom_two_thirds(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """Extract bottom 2/3 of the image"""
        height = image.shape[0]
        start_y = height // 3  # Start from 1/3 down

        cropped_image = image[start_y:, :]

        if self.verbose:
            logger.info(f"Bottom 2/3 extracted: {cropped_image.shape}")

        return cropped_image, start_y

    def detect_text_with_positions(self, image: np.ndarray) -> list:
        """Detect text and return positions using Tesseract"""
        try:
            # Configure tesseract for better Arabic/English text detection
            config = '--oem 3 --psm 6'

            # Get detailed text information with fallback
            try:
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            except Exception as e:
                logger.warning(f"Primary tesseract detection failed: {e}, trying fallback")
                # Fallback to simpler configuration
                config = '--psm 6'
                data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

            detections = []
            n_boxes = len(data['level'])

            for i in range(n_boxes):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:  # Non-empty text
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        detections.append({
                            'text': text,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'confidence': int(data['conf'][i])
                        })

            if self.verbose:
                logger.info(f"Detected {len(detections)} text elements")
                for det in detections[:5]:  # Log first 5 detections
                    logger.info(f"  Text: '{det['text']}' at ({det['x']}, {det['y']}) conf: {det['confidence']}")

            return detections

        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []

    def detect_grey_boundary(self, image: np.ndarray, start_y: int, zone_x: int, zone_width: int) -> int:
        """Detect grey boundary below the signature zone text using enhanced scanning"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)

        height = image.shape[0]

        # Enhanced scanning: look for consistent grey patterns
        for y in range(start_y + 10, min(start_y + 200, height - 10), 3):
            # Sample horizontal stripe for grey detection
            horizontal_stripe = gray_image[y-2:y+2, zone_x:zone_x+zone_width]

            if horizontal_stripe.size > 0:
                # Look for grey pixels (intensity between 100-200)
                grey_pixels = np.sum((horizontal_stripe >= 100) & (horizontal_stripe <= 200))
                total_pixels = horizontal_stripe.size
                grey_percentage = grey_pixels / total_pixels

                if grey_percentage > 0.25:  # 25% or more grey pixels (stricter threshold)
                    if self.verbose:
                        logger.info(f"Grey boundary detected at y={y}, grey%={grey_percentage:.2f}")
                    return y

        # Fallback: use larger maximum height for better signature capture
        max_height = min(200, height - start_y - 20)
        return start_y + max_height

    def refine_white_background_boundary(self, image: np.ndarray, zone_x: int, zone_width: int,
                                       initial_lower_y: int, start_y: int) -> int:
        """Scan from bottom up to ensure we have white background, shrinking zone if needed"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_lower_y = initial_lower_y
        for y in range(initial_lower_y, start_y, -5):  # Scan upward
            horizontal_stripe = gray_image[y-3:y+3, zone_x:zone_x+zone_width]

            if horizontal_stripe.size > 0:
                # Count white pixels (>= 230 intensity)
                white_pixels = np.sum(horizontal_stripe >= 230)
                total_pixels = horizontal_stripe.size
                white_percentage = white_pixels / total_pixels

                if white_percentage > 0.7:  # 70% or more is white
                    if self.verbose:
                        logger.info(f"White background confirmed at y={y}, white%={white_percentage:.2f}")
                    return y
                else:
                    current_lower_y = y - 5
                    if self.verbose:
                        logger.info(f"Non-white background at y={y}, white%={white_percentage:.2f}, shrinking")

        # Ensure minimum height
        min_height = 50
        if current_lower_y - start_y < min_height:
            current_lower_y = start_y + min_height
            if self.verbose:
                logger.info(f"Zone shrunk to minimum height: {min_height}")

        return current_lower_y

    def backup_detection_method_1(self, image: np.ndarray) -> Optional[DetectedZone]:
        """Enhanced backup method: intelligent image processing to find signature zones"""
        try:
            height, width = image.shape[:2]

            # Convert to grayscale for processing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Enhanced edge detection for finding document boundaries/signatures
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # Look for horizontal lines (potential signature zone boundaries)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

            # Find contours of horizontal lines
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                if self.verbose:
                    logger.warning("Backup method 1: No horizontal lines found")
                return None

            # Filter contours by position and size
            valid_lines = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Lines should be in lower 2/3 of document and reasonably wide
                if y > height * 0.3 and w > width * 0.2:
                    valid_lines.append({'x': x, 'y': y, 'width': w, 'height': h})

            if len(valid_lines) < 2:
                if self.verbose:
                    logger.warning(f"Backup method 1: Found only {len(valid_lines)} valid lines, need at least 2")
                return None

            # Sort by Y position
            valid_lines.sort(key=lambda line: line['y'])

            # Find two lines that could bound a signature zone
            for i in range(len(valid_lines) - 1):
                upper_line = valid_lines[i]
                lower_line = valid_lines[i + 1]

                zone_height = lower_line['y'] - upper_line['y']

                # Check if zone height is reasonable for a signature (50-200 pixels)
                if 50 <= zone_height <= 200:
                    # Use optimized method for horizontal positioning
                    base_zone_width = min(500, width - 100)  # Max 500px, leave margins
                    zone_width = int(base_zone_width * 0.8)  # 20% shrinkage (optimized)
                    zone_x = (width - zone_width) // 2  # Centered

                    # Apply 5% height reduction as in original method
                    height_reduction = int(zone_height * 0.05)
                    zone_height = zone_height - height_reduction

                    # Ensure minimum height
                    if zone_height < 50:
                        zone_height = 80

                    # Bounds checking
                    zone_start_y = max(0, min(upper_line['y'] + 10, height - zone_height))
                    zone_x = max(0, min(zone_x, width - zone_width))
                    zone_height = min(zone_height, height - zone_start_y)
                    zone_width = min(zone_width, width - zone_x)

                    if self.verbose:
                        logger.info(f"Backup method 1: Enhanced processing successful")
                        logger.info(f"  Zone: ({zone_x}, {zone_start_y}) size {zone_width}x{zone_height}")
                        logger.info(f"  Between lines at y={upper_line['y']} and y={lower_line['y']}")

                    return DetectedZone(
                        x=zone_x,
                        y=zone_start_y,
                        width=zone_width,
                        height=zone_height,
                        confidence=0.6,
                        text_above="Enhanced Image Processing"
                    )

        except Exception as e:
            if self.verbose:
                logger.warning(f"Backup detection method 1 failed: {e}")

        return None

    def backup_detection_method_2(self, image: np.ndarray) -> Optional[DetectedZone]:
        """Backup method: Grey boundary detection using original document structure analysis"""
        try:
            height, width = image.shape[:2]
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Look for horizontal grey strips that could indicate signature zone boundaries
            grey_strips = []
            min_strip_width = width * 0.3  # Strip should span at least 30% of width

            for y in range(int(height * 0.2), height - 10, 2):  # Start from 20% down
                row = gray_image[y, :]

                # Find grey pixels (intensity between 100-200)
                grey_mask = (row >= 100) & (row <= 200)
                grey_pixels_in_row = np.sum(grey_mask)

                if grey_pixels_in_row > min_strip_width:
                    # Find the span of grey pixels in this row
                    grey_positions = np.where(row > 0)[0]
                    if len(grey_positions) > 0:
                        strip_start = grey_positions[0]
                        strip_end = grey_positions[-1]
                        strip_width = strip_end - strip_start

                        # Only consider substantial horizontal strips
                        if strip_width > min_strip_width:
                            grey_strips.append({
                                'y': y,
                                'start_x': strip_start,
                                'end_x': strip_end,
                                'width': strip_width,
                                'grey_density': grey_pixels_in_row / width
                            })

            if len(grey_strips) < 2:
                if self.verbose:
                    logger.warning(f"Backup method 2: Found only {len(grey_strips)} grey strips, need at least 2")
                return None

            # Sort strips by Y position
            grey_strips.sort(key=lambda x: x['y'])

            if self.verbose:
                logger.info(f"Backup method 2: Found {len(grey_strips)} horizontal grey strips")
                for i, strip in enumerate(grey_strips[:5]):  # Log first 5
                    logger.info(f"  Strip {i}: y={strip['y']}, width={strip['width']}, density={strip['grey_density']:.2f}")

            # Find the signature zone between two prominent grey strips
            # Look for strips in the lower portion of the document
            lower_half_start = int(height * 0.4)
            candidate_pairs = []

            for i in range(len(grey_strips) - 1):
                upper_strip = grey_strips[i]
                lower_strip = grey_strips[i + 1]

                # Both strips should be in the lower portion and have good density
                if (upper_strip['y'] >= lower_half_start and
                    lower_strip['y'] >= lower_half_start and
                    upper_strip['grey_density'] > 0.2 and
                    lower_strip['grey_density'] > 0.2):

                    zone_height = lower_strip['y'] - upper_strip['y']

                    # Zone should be reasonable size (between 50 and 200 pixels)
                    if 50 <= zone_height <= 200:
                        candidate_pairs.append({
                            'upper_strip': upper_strip,
                            'lower_strip': lower_strip,
                            'zone_start_y': upper_strip['y'] + 5,  # Small offset from strip
                            'zone_height': zone_height - 10,  # Small margins
                            'confidence': min(upper_strip['grey_density'], lower_strip['grey_density'])
                        })

            if not candidate_pairs:
                if self.verbose:
                    logger.warning("Backup method 2: No suitable grey strip pairs found for signature zone")
                return None

            # Select best candidate (highest confidence)
            best_pair = max(candidate_pairs, key=lambda x: x['confidence'])

            # Use optimized method for horizontal positioning
            base_zone_width = min(500, width - 100)  # Max 500px, leave margins
            zone_width = int(base_zone_width * 0.8)  # 20% shrinkage (optimized)
            zone_x = (width - zone_width) // 2  # Centered

            # Apply 5% height reduction as in original method
            zone_height = best_pair['zone_height']
            height_reduction = int(zone_height * 0.05)
            zone_height = zone_height - height_reduction

            # Ensure minimum height
            if zone_height < 50:
                zone_height = 80

            # Bounds checking
            zone_start_y = max(0, min(best_pair['zone_start_y'], height - zone_height))
            zone_x = max(0, min(zone_x, width - zone_width))
            zone_height = min(zone_height, height - zone_start_y)
            zone_width = min(zone_width, width - zone_x)

            if self.verbose:
                logger.info(f"Backup method 2: Grey boundary detection successful")
                logger.info(f"  Zone: ({zone_x}, {zone_start_y}) size {zone_width}x{zone_height}")
                logger.info(f"  Upper strip at y={best_pair['upper_strip']['y']}, Lower strip at y={best_pair['lower_strip']['y']}")

            return DetectedZone(
                x=zone_x,
                y=zone_start_y,
                width=zone_width,
                height=zone_height,
                confidence=0.7,  # Higher confidence since it's based on actual document structure
                text_above="Grey Boundary Detection"
            )

        except Exception as e:
            if self.verbose:
                logger.warning(f"Backup detection method 2 failed: {e}")

        return None

    def build_signature_zone_from_candidate(self, image: np.ndarray, candidate: dict) -> Optional[DetectedZone]:
        """Build a signature zone from a candidate with all refinements applied"""
        try:
            zone_start_y = candidate['zone_start_y']
            zone_x = candidate['zone_x']
            zone_width = candidate['zone_width']

            # Apply the same refinement logic as the main detection
            initial_zone_end_y = self.detect_grey_boundary(image, zone_start_y, zone_x, zone_width)
            zone_end_y = self.refine_white_background_boundary(
                image, zone_x, zone_width, initial_zone_end_y, zone_start_y
            )
            zone_height = zone_end_y - zone_start_y

            # 5% height reduction
            height_reduction = int(zone_height * 0.05)
            zone_height = zone_height - height_reduction

            # Ensure minimum height (improved for better signature capture)
            if zone_height < 60:
                zone_height = min(150, image.shape[0] - zone_start_y - 20)

            # Bounds checking
            height, width = image.shape[:2]
            zone_start_y = max(0, min(zone_start_y, height - zone_height))
            zone_x = max(0, min(zone_x, width - zone_width))
            zone_height = min(zone_height, height - zone_start_y)
            zone_width = min(zone_width, width - zone_x)

            return DetectedZone(
                x=zone_x,
                y=zone_start_y,
                width=zone_width,
                height=zone_height,
                confidence=candidate['confidence'],
                text_above=candidate['text']
            )

        except Exception as e:
            if self.verbose:
                logger.warning(f"Failed to build zone from candidate: {e}")
            return None

    def _fuzzy_match(self, text: str, keyword: str) -> bool:
        """Fuzzy matching for OCR errors and variations"""
        text = text.lower().strip()
        keyword = keyword.lower().strip()
        
        # Exact match (including substring)
        if keyword in text or text in keyword:
            return True
        
        # Handle very short text (likely OCR fragments)
        if len(text) < 3:
            return False
            
        # Handle common OCR character substitutions
        ocr_replacements = {
            '0': 'o', 'o': '0', '1': 'l', 'l': 'i', 'i': 'l',
            'rn': 'm', 'm': 'rn', 'cl': 'd', 'd': 'cl',
            'u': 'o', 'a': 'e', 'e': 'a'
        }
        
        # Try with OCR corrections
        corrected_text = text
        for wrong, correct in ocr_replacements.items():
            corrected_text = corrected_text.replace(wrong, correct)
        
        if keyword in corrected_text or corrected_text in keyword:
            return True
        
        # Calculate simple edit distance for close matches
        def edit_distance(s1, s2):
            if len(s1) > len(s2):
                s1, s2 = s2, s1
            
            distances = range(len(s1) + 1)
            for index2, char2 in enumerate(s2):
                new_distances = [index2 + 1]
                for index1, char1 in enumerate(s1):
                    if char1 == char2:
                        new_distances.append(distances[index1])
                    else:
                        new_distances.append(1 + min(distances[index1], distances[index1 + 1], new_distances[-1]))
                distances = new_distances
            return distances[-1]
        
        # For longer keywords, allow some edit distance
        if len(keyword) >= 4:
            max_distance = max(1, len(keyword) // 3)  # Allow 33% character errors
            if edit_distance(text, keyword) <= max_distance:
                return True
        
        # Check if text might be a fragment of the keyword
        if len(text) >= 4 and len(keyword) >= 6:
            # Check if text is a substring with some errors
            for i in range(len(keyword) - len(text) + 1):
                if edit_distance(text, keyword[i:i+len(text)]) <= 1:
                    return True
        
        return False

    def find_signature_zone(self, image: np.ndarray, text_detections: list) -> Optional[DetectedZone]:
        """Find the signature zone based on text detection and visual cues (original method)"""
        height, width = image.shape[:2]

        # Look for specific signature-related keywords (prioritize actual signature labels)
        signature_keywords = [
            'second party\'s signature', 'party\'s signature', 'signature',
            'second party', 'signatory', 'signed', 'sign',
            'توقيع الطرف الثاني', 'توقيع', 'الطرف الثاني'  # Arabic keywords
        ]

        candidates = []
        
        # Focus search on lower portion where signatures actually appear (bottom 60%)
        signature_area_start = int(height * 0.4)  # Start from 40% down

        # Debug: Show what text is in the signature area
        if self.verbose:
            signature_area_texts = [d for d in text_detections if d['y'] >= signature_area_start]
            logger.info(f"Signature area starts at y={signature_area_start}, found {len(signature_area_texts)} text elements:")
            for i, detection in enumerate(signature_area_texts[:10]):  # Show first 10
                logger.info(f"  {i+1}. y={detection['y']}: '{detection['text']}' (conf: {detection['confidence']})")

        for detection in text_detections:
            # Skip text that appears too high in the document (contract content)
            if detection['y'] < signature_area_start:
                continue
                
            text_lower = detection['text'].lower()

            # Check if text contains signature keywords (with fuzzy matching)
            for keyword in signature_keywords:
                if self._fuzzy_match(text_lower, keyword):
                    # Calculate potential zone below this text (improved positioning)
                    zone_start_y = detection['y'] + detection['height'] + 50

                    # Estimate zone width and position (centered) - improved dimensions
                    base_zone_width = min(500, width - 100)  # Max 500px, leave margins
                    zone_x = (width - base_zone_width) // 2

                    # **REDUCED HORIZONTAL SHRINKAGE: Only 20% reduction instead of 30%**
                    zone_width = int(base_zone_width * 0.8)  # 20% reduction
                    zone_x = (width - zone_width) // 2  # Re-center after shrinkage

                    candidates.append({
                        'text': detection['text'],
                        'zone_start_y': zone_start_y,
                        'zone_x': zone_x,
                        'zone_width': zone_width,
                        'confidence': detection['confidence']
                    })

                    if self.verbose:
                        logger.info(f"Found signature keyword: '{detection['text']}' at y={detection['y']}")

        if not candidates:
            if self.verbose:
                logger.warning("No signature keywords found")
            return None

        # Select best candidate (highest confidence)
        best_candidate = max(candidates, key=lambda x: x['confidence'])
        return self.build_signature_zone_from_candidate(image, best_candidate)

    def find_signature_zone_with_backup(self, image: np.ndarray, text_detections: list) -> Optional[DetectedZone]:
        """Enhanced zone detection with backup methods when initial detection fails"""
        if self.verbose:
            logger.info("Starting signature zone detection...")

        # Primary method: text-based detection
        zone = self.find_signature_zone(image, text_detections)
        if zone:
            if self.verbose:
                logger.info("✅ Primary text-based detection successful")
            return zone

        if self.verbose:
            logger.warning("❌ Primary detection failed, trying backup methods...")

        # Backup method 1: Enhanced image processing
        zone = self.backup_detection_method_1(image)
        if zone:
            if self.verbose:
                logger.info("✅ Backup method 1 (enhanced processing) successful")
            return zone

        # Backup method 2: Grey boundary detection
        zone = self.backup_detection_method_2(image)
        if zone:
            if self.verbose:
                logger.info("✅ Backup method 2 (grey boundary) successful")
            return zone

        if self.verbose:
            logger.error("❌ All detection methods failed")
        return None

    def classify_with_gemini(self, zone_image: np.ndarray) -> GeminiClassificationResult:
        """Use Gemini to classify the zone content"""
        try:
            # Convert numpy array to PIL Image
            zone_image_rgb = cv2.cvtColor(zone_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(zone_image_rgb)

            # Create the prompt for classification
            prompt = """
            You are an expert document analyzer. Please analyze this image which represents a signature zone from an official contract document.

            Your task is to classify this zone into one of exactly three categories:

            1. **"signature"** - If the zone contains handwritten signatures, initials, or any form of handwriting/pen marks
            2. **"contract"** - If the zone contains a document image, labor card, ID card, passport page, or any rectangular/page-like document or photo of a person
            3. **"blank"** - If the zone is mostly empty/white with no significant content

            **IMPORTANT CLASSIFICATION RULES:**
            - If you see ANY handwriting, signatures, or pen marks → classify as "signature"
            - If you see a rectangular document, card, photo, or any embedded page/image → classify as "contract"
            - Only classify as "blank" if the area is genuinely empty or contains only printed text/borders
            - Be very careful to distinguish between handwritten content (signature) and printed/embedded documents (contract)

            Please respond in this exact JSON format:
            {
                "classification": "signature|contract|blank",
                "confidence": 0.0-1.0,
                "reasoning": "Detailed explanation of what you see and why you classified it this way"
            }

            Analyze the image carefully and provide your classification.
            """

            response = self.model.generate_content([prompt, pil_image])

            if self.verbose:
                logger.info(f"Gemini raw response: {response.text}")

            # Parse the JSON response
            try:
                # Extract JSON from response text
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    result_data = json.loads(json_str)
                else:
                    # Fallback parsing if JSON is not properly formatted
                    raise ValueError("No valid JSON found in response")

                # Validate the classification
                valid_classifications = ["signature", "contract", "blank"]
                classification = result_data.get("classification", "").lower()

                if classification not in valid_classifications:
                    logger.warning(f"Invalid classification '{classification}', defaulting to 'blank'")
                    classification = "blank"

                confidence = float(result_data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1

                reasoning = result_data.get("reasoning", "Gemini analysis completed")

                return GeminiClassificationResult(
                    classification=classification,
                    confidence=confidence,
                    reasoning=reasoning,
                    gemini_response=response.text
                )

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing Gemini response: {e}")
                logger.error(f"Raw response: {response.text}")

                # Fallback classification based on text analysis
                response_lower = response.text.lower()
                if "signature" in response_lower or "handwriting" in response_lower:
                    classification = "signature"
                    confidence = 0.6
                elif "contract" in response_lower or "document" in response_lower:
                    classification = "contract"
                    confidence = 0.6
                else:
                    classification = "blank"
                    confidence = 0.5

                return GeminiClassificationResult(
                    classification=classification,
                    confidence=confidence,
                    reasoning=f"Fallback analysis due to parsing error: {e}",
                    gemini_response=response.text
                )

        except Exception as e:
            logger.error(f"Gemini classification failed: {e}")
            return GeminiClassificationResult(
                classification="error",
                confidence=0.0,
                reasoning=f"Classification failed: {e}",
                gemini_response=""
            )

    def process_contract_for_signature(self, file_content: bytes, filename: str) -> Optional[np.ndarray]:
        """Main processing pipeline for signature extraction from uploaded file"""
        logger.info(f"Processing contract: {filename}")

        try:
            # Step 1: Extract image from uploaded file
            original_image = self.extract_image_from_upload(file_content, filename)

            # Step 2: Get bottom 2/3
            cropped_image, start_y = self.get_bottom_two_thirds(original_image)

            # Step 3: Detect text with positions
            text_detections = self.detect_text_with_positions(cropped_image)

            # Step 4: Find signature zone
            signature_zone = self.find_signature_zone_with_backup(cropped_image, text_detections)

            if signature_zone:
                # Step 5: Extract the signature zone
                zone_image = cropped_image[signature_zone.y:signature_zone.y+signature_zone.height,
                                         signature_zone.x:signature_zone.x+signature_zone.width]
                
                if self.verbose:
                    logger.info(f"✅ Signature zone extracted: {zone_image.shape}")
                    logger.info(f"Zone Location: ({signature_zone.x}, {signature_zone.y})")
                    logger.info(f"Zone Size: {signature_zone.width}x{signature_zone.height}")
                
                return zone_image
            else:
                logger.warning("❌ No signature zone detected")
                return None

        except Exception as e:
            logger.error(f"Error processing contract: {e}")
            return None

# Initialize FastAPI app
app = FastAPI(title="Signature Cropper API", description="API for extracting signatures from contract documents")

# Initialize the classifier with API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDmNcG-gz9PN4zmoq5anCstMqdfhXqJaL0")
classifier = GeminiZoneClassifier(GEMINI_API_KEY, verbose=True)

@app.post("/crop-signature")
async def crop_signature(file: UploadFile = File(...)):
    """
    Extract and return the cropped signature from a contract document.
    
    Accepts PDF files or image files and returns the cropped signature as JPG.
    """
    # Validate file type
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process the contract to extract signature
        signature_image = classifier.process_contract_for_signature(file_content, file.filename)
        
        if signature_image is None:
            raise HTTPException(
                status_code=404,
                detail="No signature zone could be detected in the provided contract"
            )
        
        # Convert signature image to JPG format
        # Convert BGR to RGB for PIL
        signature_image_rgb = cv2.cvtColor(signature_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(signature_image_rgb)
        
        # Save to bytes buffer as JPG
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        # Return the image as response
        return Response(
            content=img_buffer.getvalue(),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=signature_{Path(file.filename).stem}.jpg"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Signature Cropper API is running", "status": "healthy"}

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini_configured": GEMINI_API_KEY is not None,
        "tesseract_available": classifier.tesseract_path is not None
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
