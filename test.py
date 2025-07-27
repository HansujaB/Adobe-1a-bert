import os
from main import PDFOutlineExtractor  # Assuming your main code is saved as extractor.py
import json

def test_batch_pdf_outline(input_folder="input", output_folder="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load extractor
    extractor = PDFOutlineExtractor()

    # Process all PDFs in input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        json_filename = os.path.splitext(pdf_file)[0] + ".json"
        output_path = os.path.join(output_folder, json_filename)

        print(f"Processing: {pdf_file}")

        try:
            result = extractor.extract_outline(pdf_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved to {output_path}")

        except Exception as e:
            print(f"❌ Error processing {pdf_file}: {e}")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"title": "", "outline": []}, f, indent=2)

if __name__ == "__main__":
    test_batch_pdf_outline()
