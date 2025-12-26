import fitz  # PyMuPDF
import re
import csv
import argparse
from tqdm import tqdm

def clean_text_for_csv(parts):
    """Joins parts and removes CSV-breaking characters."""
    full_text = " ".join(parts)
    # Replace newlines, tabs, and carriage returns with a single space
    return re.sub(r"[\r\n\t]+", " ", full_text).strip()

def main():

    # CLI args.
    parser = argparse.ArgumentParser(description="Load and process a PDF file.")
    parser.add_argument("pdf_path", type=str, help="Path to the input PDF file.")
    args = parser.parse_args()

    # Load the PDF
    doc = fitz.open(args.pdf_path)

    # Constants. 
    ALLOWED_CHARS = r"[^a-zāčēģīķļņšūž\-–\[\]]"
    ALLOWED_CHARS_RE = re.compile(ALLOWED_CHARS, re.IGNORECASE)
    PAGE_OFFSET = -3
    RANGE_START = 57
    RANGE_END = 1219
    INDENT_THRESHOLD = 4
    page_range = range(RANGE_START - 1, RANGE_END)

    # Manual marking of pages where there is no headwords (entries).
    PAGE_WITH_NO_HEADWORDS = [
      117, 172, 271, 297, 334, 397, 472, 492, 493, 499, 510, 535, 582, 655, 
      670, 689, 743, 808, 975, 1009, 1025, 1043, 1049, 1092, 1157, 1214
    ]
    PAGE_WITH_NO_HEADWORDS = [p - 1 for p in PAGE_WITH_NO_HEADWORDS]

    words = set()
    current_entry = {"headword": "", "text_parts": [], "page": 1 + PAGE_OFFSET}
    row_number = 1
    
    with (
        open("entry_raw_data.csv", "w", encoding="utf-8", newline='') as csv_file,
        open("temp_results.txt", "w", encoding="utf-8") as txt_file
    ):
        # QUOTE_MINIMAL is standard, but QUOTE_ALL ensures maximum safety for text fields
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["row_number", "headword", "page", "text"])

        for page_num in tqdm(page_range, desc="Processing pages"):

            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]

            # 1. Collect X positions
            positions = []
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        positions.append(span["bbox"][0])

            # 2. Determine indentation levels
            positions_sorted = sorted(set(positions))
            indent_levels = [positions_sorted[0]] if positions_sorted else []
            for pos in positions_sorted[1:]:
                if all(abs(pos - level) >= INDENT_THRESHOLD for level in indent_levels):
                    indent_levels.append(pos)
                if len(indent_levels) >= 3:
                    break

            # 3. Process Text
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):

                        text = span["text"].strip()
                        x0 = span["bbox"][0]

                        # Skip logic for image pages
                        if page_num in PAGE_WITH_NO_HEADWORDS:
                            current_entry["text_parts"].append(text)
                            continue

                        # Determine indent
                        indent_level = None
                        for i, lvl in enumerate(indent_levels):
                            if abs(x0 - lvl) < INDENT_THRESHOLD:
                                indent_level = i
                                break

                        is_headword = indent_level == 0 and len(text) > 1
                        
                        if is_headword:
                            # Clean headword
                            if ALLOWED_CHARS_RE.search(text):
                                cleaned_text = ALLOWED_CHARS_RE.sub("", text)
                            else:
                                cleaned_text = text

                            # FLUSH PREVIOUS ENTRY
                            if current_entry["headword"]:
                                # Sanitize text for CSV validity
                                safe_text = clean_text_for_csv(current_entry["text_parts"])
                                
                                csv_writer.writerow([
                                    row_number,
                                    current_entry["headword"],
                                    current_entry["page"], 
                                    safe_text
                                ])

                                txt_file.write(f"{current_entry['headword']}: {safe_text}\n")
                                row_number += 1

                            # Check for duplicates
                            if cleaned_text in words:
                                current_entry = {
                                    "headword": "", 
                                    "text_parts": [],
                                    "page": page_num + 1 + PAGE_OFFSET
                                }
                                continue
                            
                            words.add(cleaned_text)

                            # Start new entry
                            current_entry = {
                                "headword": cleaned_text,
                                "text_parts": [],
                                "page": page_num + 1 + PAGE_OFFSET
                            }

                        else:
                            current_entry["text_parts"].append(text)

        # Flush final entry
        if current_entry["headword"]:
            safe_text = clean_text_for_csv(current_entry["text_parts"])
            csv_writer.writerow([
                row_number,
                current_entry["headword"],
                current_entry["page"],
                safe_text
            ])

if __name__ == "__main__":
    main()
