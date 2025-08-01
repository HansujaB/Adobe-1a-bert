#!/usr/bin/env python3
"""
Enhanced PDF Outline Extractor for Adobe India Hackathon Round 1A
Extracts structured outlines (Title, H1, H2, H3) from PDF documents
"""

import os
import json
import re
import fitz  # PyMuPDF
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from datetime import datetime
import argparse
import logging
import torch
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def __init__(self, model_path: str = "models/bert-mini"):
        """Initialize the PDF outline extractor with a pre-trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model with custom classification head
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,  # Title, H1, H2, H3, Other
            ignore_mismatched_sizes=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping
        self.label_map = {0: "Title", 1: "H1", 2: "H2", 3: "H3", 4: "Other"}
        
        # Heuristic patterns for better classification
        self.title_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s*[A-Z]',    # Numbered sections
            r'^Chapter\s+\d+',    # Chapter headings
            r'^CHAPTER\s+\d+',    # ALL CAPS chapter
        ]
        
        self.heading_patterns = [
            r'^\d+\.\d+\s',       # 1.1 format
            r'^\d+\.\d+\.\d+\s',  # 1.1.1 format
            r'^[A-Z]\.\s',        # A. format
            r'^\([a-z]\)\s',      # (a) format
        ]
        
    def extract_text_blocks(self, pdf_path: str) -> List[Dict]:
        """Extract text blocks with enhanced metadata from PDF."""
        doc = fitz.open(pdf_path)
        blocks = []
        
        # Global font statistics for better normalization
        all_font_sizes = []
        all_blocks_raw = []
        
        # First pass: collect all font sizes
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            all_font_sizes.append(span["size"])
        
        # Calculate global font percentiles
        if all_font_sizes:
            font_percentiles = np.percentile(all_font_sizes, [10, 25, 50, 75, 85, 95])
            median_font = np.median(all_font_sizes)
        else:
            font_percentiles = [8, 10, 12, 14, 16, 18]
            median_font = 12
            
        # Second pass: extract blocks with enhanced features
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            page_height = page.rect.height
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        font_sizes = []
                        font_flags = []
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            font_sizes.append(span["size"])
                            font_flags.append(span["flags"])
                        
                        # Clean text
                        line_text = line_text.strip()
                        line_text = re.sub(r'\s+', ' ', line_text)  # Normalize whitespace
                        
                        # Skip invalid text
                        if (len(line_text) < 3 or len(line_text) > 500 or 
                            not re.search(r'[a-zA-Z]', line_text)):
                            continue
                        
                        # Skip page numbers, references, etc.
                        if (re.match(r'^\d+$', line_text) or 
                            re.match(r'^Page\s+\d+', line_text, re.IGNORECASE) or
                            len(re.findall(r'[0-9]', line_text)) > len(line_text) * 0.7):
                            continue
                        
                        # Calculate font statistics
                        avg_font_size = np.mean(font_sizes) if font_sizes else 12
                        max_font_size = max(font_sizes) if font_sizes else 12
                        
                        # Font style flags (bold, italic)
                        is_bold = any(flag & 16 for flag in font_flags)  # Bold flag
                        is_italic = any(flag & 2 for flag in font_flags)  # Italic flag
                        
                        # Position features
                        bbox = line["bbox"]
                        y_pos_norm = bbox[1] / page_height  # Normalized y position
                        
                        # Calculate font percentile
                        font_percentile = np.searchsorted(font_percentiles, avg_font_size) * 100 / len(font_percentiles)
                        
                        # Text features
                        word_count = len(line_text.split())
                        has_colon = ':' in line_text
                        starts_with_number = bool(re.match(r'^\d+\.', line_text))
                        is_question = line_text.endswith('?')
                        
                        blocks.append({
                            "text": line_text,
                            "page": page_num + 1,
                            "font_size": avg_font_size,
                            "max_font_size": max_font_size,
                            "font_percentile": min(font_percentile, 100),
                            "font_ratio": avg_font_size / median_font,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "bbox": bbox,
                            "y_position": bbox[1],
                            "y_pos_norm": y_pos_norm,
                            "word_count": word_count,
                            "has_colon": has_colon,
                            "starts_with_number": starts_with_number,
                            "is_question": is_question,
                            "is_title_case": line_text.istitle(),
                            "is_upper": line_text.isupper(),
                            "char_count": len(line_text)
                        })
        
        doc.close()
        logger.info(f"Extracted {len(blocks)} text blocks")
        return blocks
    
    def apply_heuristic_rules(self, blocks: List[Dict]) -> List[Dict]:
        """Apply heuristic rules before ML classification."""
        for block in blocks:
            text = block["text"]
            
            # Title heuristics
            if (block["page"] <= 2 and 
                (block["font_ratio"] > 1.5 or block["font_percentile"] > 85) and
                block["word_count"] <= 15 and
                not text.endswith('.') and
                not block["starts_with_number"]):
                block["heuristic_hint"] = "Title"
                
            # H1 heuristics  
            elif (block["font_ratio"] > 1.2 and
                  (block["is_bold"] or block["font_percentile"] > 70) and
                  block["word_count"] <= 20):
                block["heuristic_hint"] = "H1"
                
            # H2/H3 heuristics based on numbering
            elif any(re.match(pattern, text) for pattern in self.heading_patterns):
                if re.match(r'^\d+\.\d+\.\d+', text):
                    block["heuristic_hint"] = "H3"
                elif re.match(r'^\d+\.\d+', text):
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
            
            feature_text = f"{font_tag} {position_tag} {style_tag} {content_tag} {hint_tag} {text}"
            texts.append(feature_text)
        
        # ML Classification
        predictions = []
        confidences = []
        batch_size = 16  # Increased batch size for better efficiency
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,  # Increased for richer features
                return_tensors="pt"
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                batch_predictions = torch.argmax(probabilities, dim=-1).cpu().numpy()
                batch_confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
                
            predictions.extend(batch_predictions)
            confidences.extend(batch_confidences)
        
        # Combine ML predictions with heuristics
        for i, block in enumerate(blocks):
            ml_prediction = self.label_map[predictions[i]]
            heuristic_prediction = block["heuristic_hint"]
            confidence = float(confidences[i])
            
            # Use heuristics for high-confidence cases
            if (heuristic_prediction != "Other" and 
                (confidence < 0.7 or heuristic_prediction == "Title")):
                final_prediction = heuristic_prediction
                confidence = max(confidence, 0.8)  # Boost confidence
            else:
                final_prediction = ml_prediction
                
            block["predicted_label"] = final_prediction
            block["confidence"] = confidence
            block["ml_prediction"] = ml_prediction
        
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
    parser.add_argument("--input", default="/app/input", help="Input directory containing PDFs")
    parser.add_argument("--output", default="/app/output", help="Output directory for JSON files")
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