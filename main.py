#!/usr/bin/env python3
"""
Enhanced PDF Outline Extractor for Adobe India Hackathon Round 1A

This module provides functionality to extract structured outlines from PDF documents
using BERT-based machine learning models. It identifies titles, headings (H1, H2, H3),
and creates hierarchical document structures.

Features:
- PDF text extraction with font and formatting analysis
- BERT-based text classification for heading detection
- Heuristic pattern matching for improved accuracy
- Hierarchical outline generation with confidence scores
- Batch processing of multiple PDF files

Author: Adobe India Hackathon Participant
Date: 2025
"""

# Standard library imports
import os
import json
import re
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

# Third-party imports
import fitz  # PyMuPDF - PDF processing library
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging to provide detailed processing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    """
    A comprehensive PDF outline extraction system using BERT-based classification.
    
    This class combines machine learning with heuristic approaches to identify
    and classify document structure elements like titles and headings from PDF files.
    
    Attributes:
        device (torch.device): Computation device (CPU/GPU)
        tokenizer: BERT tokenizer for text preprocessing
        model: Pre-trained BERT model for sequence classification
        label_map (dict): Mapping from numeric labels to heading types
        title_patterns (list): Regex patterns for title detection
        heading_patterns (list): Regex patterns for heading detection
    """
    
    def __init__(self, model_path: str = "models/bert-mini"):
        """
        Initialize the PDF outline extractor with a pre-trained BERT model.
        
        Args:
            model_path (str): Path to the pre-trained model directory or HuggingFace model name
        """
        # Determine computation device (prefer GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer for text preprocessing
        # The tokenizer converts raw text into tokens that the model can understand
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load pre-trained BERT model with custom classification head
        # num_labels=5 corresponds to: Title, H1, H2, H3, Other
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,  # Classification categories for document structure
            ignore_mismatched_sizes=True  # Allow model architecture modifications
        )
        self.model.to(self.device)  # Move model to appropriate device
        self.model.eval()  # Set model to evaluation mode (disable dropout, etc.)
        
        # Define mapping from model output indices to semantic labels
        self.label_map = {
            0: "Title",  # Document title
            1: "H1",     # Main headings
            2: "H2",     # Sub-headings
            3: "H3",     # Sub-sub-headings
            4: "Other"   # Regular content
        }
        
        # Heuristic regex patterns for enhanced title detection
        # These patterns catch common title formatting conventions
        self.title_patterns = [
            r'^[A-Z][A-Z\s]+$',      # ALL CAPS titles
            r'^\d+\.\s*[A-Z]',       # Numbered sections (1. Introduction)
            r'^Chapter\s+\d+',       # Chapter headings
            r'^CHAPTER\s+\d+',       # ALL CAPS chapters
            r'^[A-Z][^.!?]*$',       # Capitalized sentences without punctuation
        ]
        
        # Heuristic patterns for hierarchical heading detection
        # These help identify structured numbering systems
        self.heading_patterns = [
            r'^\d+\.\d+\s',          # Decimal numbering (1.1, 2.3)
            r'^\d+\.\d+\.\d+\s',     # Triple decimal (1.1.1, 2.3.4)
            r'^[A-Z]\.\s',           # Letter enumeration (A. B. C.)
            r'^\([a-z]\)\s',         # Parenthetical enumeration (a) b) c)
            r'^\([0-9]+\)\s',        # Numbered parentheses (1) 2) 3)
        ]
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """
        Extract text blocks with comprehensive metadata from PDF pages.
        
        This method performs a two-pass analysis:
        1. First pass: Collect global font statistics for normalization
        2. Second pass: Extract text blocks with enhanced formatting features
        
        Args:
            pdf_path (str): Path to the PDF file to process
            
        Returns:
            List[Dict]: List of text blocks with metadata including:
                - text: The actual text content
                - page: Page number (0-indexed)
                - font_size: Font size of the text
                - font_flags: Formatting flags (bold, italic)
                - bbox: Bounding box coordinates
                - position_score: Normalized position on page
                - font_size_percentile: Font size relative to document
        """
        doc = fitz.open(pdf_path)
        blocks = []
        
        # Collect global font statistics for better normalization
        # This helps identify relative importance of text based on font size
        all_font_sizes = []
        all_blocks_raw = []
        
        # First pass: Analyze font distribution across entire document
        # This gives us context for what constitutes "large" or "small" text
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")  # Extract structured text data
            
            # Iterate through all text blocks on the page
            for block in text_dict["blocks"]:
                if "lines" in block:  # Only process text blocks (not images)
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_font_sizes.append(span["size"])
        
        # Calculate statistical measures for font size normalization
        # These percentiles help classify text importance
        if all_font_sizes:
            font_percentiles = np.percentile(all_font_sizes, [10, 25, 50, 75, 85, 95])
            median_font = np.median(all_font_sizes)
        else:
            # Fallback values if no fonts detected
            font_percentiles = [8, 10, 12, 14, 16, 18]
            median_font = 12
            
        # Second pass: Extract blocks with comprehensive feature analysis
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            page_height = page.rect.height  # For position normalization
            
            # Process each text block on the page
            for block in text_dict["blocks"]:
                if "lines" in block:  # Skip image blocks
                    for line in block["lines"]:
                        line_text = ""
                        # Collect font and formatting information from each text span
                        font_sizes = []
                        font_flags = []
                        
                        # Aggregate text and formatting data from all spans in the line
                        for span in line["spans"]:
                            line_text += span["text"]
                            font_sizes.append(span["size"])
                            font_flags.append(span["flags"])
                        
                        # Clean and normalize text content
                        line_text = line_text.strip()
                        line_text = re.sub(r'\s+', ' ', line_text)  # Normalize whitespace
                        
                        # Filter out invalid or unwanted text blocks
                        if (len(line_text) < 3 or len(line_text) > 500 or 
                            not re.search(r'[a-zA-Z]', line_text)):
                            continue  # Skip too short, too long, or non-alphabetic text
                        
                        # Skip common PDF artifacts (page numbers, references, etc.)
                        if (re.match(r'^\d+$', line_text) or 
                            re.match(r'^Page\s+\d+', line_text, re.IGNORECASE) or
                            len(re.findall(r'[0-9]', line_text)) > len(line_text) * 0.7):
                            continue  # Skip if mostly numeric
                        
                        # Calculate statistical font properties
                        avg_font_size = np.mean(font_sizes) if font_sizes else 12
                        max_font_size = max(font_sizes) if font_sizes else 12
                        
                        # Extract font style information from flags
                        # PyMuPDF uses bit flags: 16=bold, 2=italic
                        is_bold = any(flag & 16 for flag in font_flags)
                        is_italic = any(flag & 2 for flag in font_flags)
                        
                        # Calculate position-based features
                        bbox = line["bbox"]  # Bounding box [x0, y0, x1, y1]
                        y_pos_norm = bbox[1] / page_height  # Normalized y position (0-1)
                        
                        # Determine font size percentile relative to document
                        font_percentile = np.searchsorted(font_percentiles, avg_font_size) * 100 / len(font_percentiles)
                        
                        # Extract content-based text features
                        word_count = len(line_text.split())
                        has_colon = ':' in line_text  # Often indicates headings
                        starts_with_number = bool(re.match(r'^\d+\.', line_text))  # Numbered sections
                        is_question = line_text.endswith('?')
                        
                        # Create comprehensive text block metadata
                        blocks.append({
                            "text": line_text,
                            "page": page_num + 1,  # 1-indexed page number
                            "font_size": avg_font_size,
                            "max_font_size": max_font_size,
                            "font_percentile": min(font_percentile, 100),
                            "font_ratio": avg_font_size / median_font,  # Relative font size
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "bbox": bbox,  # Bounding box coordinates
                            "y_position": bbox[1],  # Absolute y position
                            "y_pos_norm": y_pos_norm,  # Normalized y position
                            "word_count": word_count,
                            "has_colon": has_colon,
                            "starts_with_number": starts_with_number,
                            "is_question": is_question,
                            "is_title_case": line_text.istitle(),  # Title Case Text
                            "is_upper": line_text.isupper(),  # ALL CAPS
                            "char_count": len(line_text)
                        })
        
        doc.close()
        logger.info(f"Extracted {len(blocks)} text blocks")
        return blocks
    
    def apply_heuristic_rules(self, blocks: List[Dict]) -> List[Dict]:
        """
        Apply rule-based heuristics to enhance text classification.
        
        This method uses predefined patterns and formatting rules to identify
        likely titles and headings before machine learning classification.
        Heuristics often catch formatting conventions that ML might miss.
        
        Args:
            blocks (List[Dict]): List of text blocks with metadata
            
        Returns:
            List[Dict]: Text blocks with added heuristic classifications
        """
        for block in blocks:
            text = block["text"]
            
            # Initialize heuristic classification scores
            block["heuristic_title"] = False
            block["heuristic_heading"] = False
            
            # Title detection heuristics - look for common title patterns
            # Titles are typically: early in document, large font, short, no ending period
            if (block["page"] <= 2 and 
                (block["font_ratio"] > 1.5 or block["font_percentile"] > 85) and
                block["word_count"] <= 15 and
                not text.endswith('.') and
                not block["starts_with_number"]):
                block["heuristic_hint"] = "Title"
                
            # H1 heuristics - main headings detection
            # Look for larger fonts, bold text, reasonable length
            elif (block["font_ratio"] > 1.2 and
                  (block["is_bold"] or block["font_percentile"] > 70) and
                  block["word_count"] <= 20):
                block["heuristic_hint"] = "H1"
                
            # H2/H3 hierarchical classification based on numbering patterns
            # Use regex patterns to identify structured numbering systems
            elif any(re.match(pattern, text) for pattern in self.heading_patterns):
                if re.match(r'^\d+\.\d+\.\d+', text):  # Triple decimal = H3
                    block["heuristic_hint"] = "H3"
                elif re.match(r'^\d+\.\d+', text):  # Double decimal = H2
                    block["heuristic_hint"] = "H2"
                else:
                    block["heuristic_hint"] = "H2"
            else:
                block["heuristic_hint"] = "Other"
                
        return blocks
    
    def classify_text_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """Enhanced classification using both ML and heuristics."""
        if not blocks:
            return []
        
        # Apply heuristic rules first
        blocks = self.apply_heuristic_rules(blocks)
        
        # Prepare enhanced features for ML classification
        texts = []
        
        for block in blocks:
            # Create rich feature representation
            text = block["text"]
            
            # Feature tags
            font_tag = f"[FONT_RATIO:{block['font_ratio']:.2f}]"
            position_tag = f"[PAGE:{block['page']},Y:{block['y_pos_norm']:.2f}]"
            style_tag = f"[BOLD:{block['is_bold']},ITALIC:{block['is_italic']}]"
            content_tag = f"[WORDS:{block['word_count']},CAPS:{block['is_upper']}]"
            hint_tag = f"[HINT:{block['heuristic_hint']}]"
            
            # Combine all features into a single input for BERT
            feature_text = f"{font_tag} {position_tag} {style_tag} {content_tag} {hint_tag} {text}"
            texts.append(feature_text)
        
        # Step 3: Perform ML Classification using BERT model
        predictions = []
        confidences = []
        batch_size = 16  # Process multiple texts simultaneously for efficiency
        
        # Process texts in batches to optimize GPU memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch of texts for BERT input
            # truncation=True ensures texts fit model's max length
            # padding=True makes all sequences same length for batch processing
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,  # BERT's maximum sequence length
                return_tensors="pt"  # Return PyTorch tensors
            ).to(self.device)
            
            # Run inference without gradient computation (faster)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Convert logits to probabilities using softmax
                probabilities = torch.softmax(outputs.logits, dim=-1)
                # Get predicted class (highest probability)
                batch_predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()
                # Get confidence scores (max probability)
                batch_confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
                
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
        
        # Step 4: Combine ML predictions with heuristic insights
        # This hybrid approach leverages both pattern-based rules and ML learning
        for i, block in enumerate(blocks):
            ml_prediction = self.label_map[predictions[i]]  # Convert index to label
            heuristic_prediction = block["heuristic_hint"]
            confidence = float(confidences[i])
            
            # Decision logic: prefer heuristics for certain cases
            # - When ML confidence is low (< 0.7)
            # - When heuristics detect titles (often more reliable)
            # - When heuristics provide non-"Other" classifications
            if (heuristic_prediction != "Other" and 
                (confidence < 0.7 or heuristic_prediction == "Title")):
                final_prediction = heuristic_prediction
                confidence = max(confidence, 0.8)  # Boost confidence for heuristic matches
            else:
                final_prediction = ml_prediction  # Trust ML for high-confidence predictions
                
            # Store final classification results
            block["predicted_label"] = final_prediction
            block["confidence"] = confidence
            block["ml_prediction"] = ml_prediction  # Keep original ML prediction for analysis
        
        return blocks
    
    def post_process_predictions(self, blocks: List[Dict]) -> Dict:
        """Enhanced post-processing with better hierarchy detection."""
        if not blocks:
            return {"title": "", "outline": []}
        
        # Sort by page and position
        blocks.sort(key=lambda x: (x["page"], x["y_position"]))
        
        # Title detection with multiple strategies
        title_candidates = []
        
        # Strategy 1: Look for predicted titles
        predicted_titles = [b for b in blocks if b["predicted_label"] == "Title"]
        title_candidates.extend(predicted_titles)
        
        # Strategy 2: Look for large font text on first few pages
        early_blocks = [b for b in blocks if b["page"] <= 3]
        if early_blocks:
            font_threshold = np.percentile([b["font_size"] for b in early_blocks], 90)
            large_font_blocks = [b for b in early_blocks 
                               if b["font_size"] >= font_threshold and 
                                  b["word_count"] <= 15 and
                                  not b["text"].endswith('.')]
            title_candidates.extend(large_font_blocks)
        
        # Select best title
        title = ""
        if title_candidates:
            # Prefer early page, large font, short text
            title_candidate = max(title_candidates, 
                                key=lambda x: (
                                    x["page"] <= 2,  # Prefer first 2 pages
                                    x["font_ratio"],
                                    -x["y_position"],  # Higher on page
                                    x["confidence"]
                                ))
            title = title_candidate["text"]
        
        # Extract and organize outline
        outline = []
        heading_levels = ["H1", "H2", "H3"]
        
        # Apply confidence thresholds by level
        confidence_thresholds = {"H1": 0.4, "H2": 0.35, "H3": 0.3}
        
        for block in blocks:
            if block["predicted_label"] in heading_levels:
                threshold = confidence_thresholds[block["predicted_label"]]
                
                if block["confidence"] >= threshold:
                    outline.append({
                        "level": block["predicted_label"],
                        "text": block["text"],
                        "page": block["page"],
                        "confidence": round(block["confidence"], 2)
                    })
        
        # Remove duplicates and clean up
        seen = set()
        unique_outline = []
        
        for item in outline:
            # Create a more flexible key for duplicate detection
            clean_text = re.sub(r'^\d+\.?\s*', '', item["text"])  # Remove numbering
            key = (item["level"], clean_text.lower().strip())
            
            if key not in seen and len(item["text"]) >= 3:
                seen.add(key)
                unique_outline.append(item)
        
        # Ensure proper hierarchy
        unique_outline = self.fix_hierarchy(unique_outline)
        
        return {
            "title": title,
            "outline": unique_outline
        }
    
    def fix_hierarchy(self, outline: List[Dict]) -> List[Dict]:
        """Ensure proper hierarchical structure (H1 -> H2 -> H3)."""
        if not outline:
            return outline
            
        fixed_outline = []
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        
        for item in outline:
            level = item["level"]
            
            # If we see H2 without H1, or H3 without H1/H2, adjust
            if level == "H2" and level_counts["H1"] == 0:
                item["level"] = "H1"
                level = "H1"
            elif level == "H3" and level_counts["H1"] == 0:
                item["level"] = "H1" 
                level = "H1"
            elif level == "H3" and level_counts["H2"] == 0:
                item["level"] = "H2"
                level = "H2"
                
            level_counts[level] += 1
            fixed_outline.append(item)
        
        return fixed_outline
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main method to extract outline from PDF."""
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text blocks
            blocks = self.extract_text_blocks(pdf_path)
            
            if not blocks:
                logger.warning("No text blocks extracted")
                return {"title": "", "outline": []}
            
            # Classify blocks
            classified_blocks = self.classify_text_blocks(blocks)
            
            # Post-process predictions
            result = self.post_process_predictions(classified_blocks)
            
            logger.info(f"Found title: '{result['title'][:50]}...' (truncated)")
            logger.info(f"Found {len(result['outline'])} headings")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"title": "", "outline": []}

def process_directory(input_dir: str, output_dir: str, model_path: str = "models/bert-mini"):
    """Process all PDFs in input directory and save results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = PDFOutlineExtractor(model_path)
    
    # Process all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    results_summary = []
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(pdf_file)[0]}.json")
        
        try:
            # Extract outline
            result = extractor.extract_outline(pdf_path)
            
            # Add metadata
            result["metadata"] = {
                "filename": pdf_file,
                "processing_timestamp": datetime.now().isoformat(),
                "outline_count": len(result["outline"])
            }
            
            # Save result
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            results_summary.append({
                "file": pdf_file,
                "status": "success",
                "title_found": bool(result["title"]),
                "outline_count": len(result["outline"])
            })
            
            logger.info(f"✅ Processed {pdf_file} -> {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_file}: {str(e)}")
            
            # Create empty result for failed files
            empty_result = {
                "title": "",
                "outline": [],
                "metadata": {
                    "filename": pdf_file,
                    "processing_timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(empty_result, f, indent=2)
                
            results_summary.append({
                "file": pdf_file,
                "status": "error",
                "error": str(e)
            })
    
    # Save processing summary
    summary_file = os.path.join(output_dir, "_processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_files": len(pdf_files),
            "successful": len([r for r in results_summary if r["status"] == "success"]),
            "failed": len([r for r in results_summary if r["status"] == "error"]),
            "results": results_summary,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    logger.info(f"Processing complete! Summary saved to {summary_file}")

def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(description="Extract structured outlines from PDFs")
    parser.add_argument("--input", default="input", help="Input directory containing PDFs")
    parser.add_argument("--output", default="output", help="Output directory for JSON files")
    parser.add_argument("--model", default="prajjwal1/bert-mini", help="Model path")
    
    args = parser.parse_args()
    
    logger.info(f"Starting PDF outline extraction")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Model: {args.model}")
    
    if not os.path.exists(args.input):
        logger.error(f"Input directory does not exist: {args.input}")
        return
    
    process_directory(args.input, args.output, args.model)
    
    logger.info("All processing complete!")

if __name__ == "__main__":
    main()