#!/usr/bin/env python3
"""
Test script to validate PDF outline extraction results
"""

import os
import json
import sys
from jsonschema import validate

def validate_output_format(output_dir):
    """Validate that all JSON files in the output directory have the correct format."""
    
    # Define the expected JSON schema
    schema = {
        "type": "object",
        "required": ["title", "outline"],
        "properties": {
            "title": {"type": "string"},
            "outline": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["level", "text", "page"],
                    "properties": {
                        "level": {"type": "string", "enum": ["H1", "H2", "H3", "H4"]},
                        "text": {"type": "string"},
                        "page": {"type": "integer"}
                    }
                }
            }
        }
    }
    
    # Get all JSON files in the output directory
    json_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and f != '_processing_summary.json']
    
    if not json_files:
        print("❌ No JSON files found in the output directory.")
        return False
    
    all_valid = True
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Validate against schema
            validate(instance=data, schema=schema)
            
            # Additional checks
            if not data["title"] and not data["outline"]:
                print(f"⚠️ {json_file}: Empty title and outline.")
            else:
                print(f"✅ {json_file}: Valid format.")
                
        except json.JSONDecodeError:
            print(f"❌ {json_file}: Invalid JSON format.")
            all_valid = False
        except Exception as e:
            print(f"❌ {json_file}: {str(e)}")
            all_valid = False
    
    return all_valid

def main():
    """Main function."""
    output_dir = "output"
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"Validating JSON files in '{output_dir}'...")
    
    if validate_output_format(output_dir):
        print("✅ All files passed validation.")
        return 0
    else:
        print("❌ Some files failed validation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
