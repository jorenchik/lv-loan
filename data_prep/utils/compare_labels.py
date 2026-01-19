import argparse
import csv
import sys

def get_lemmas_from_csv(filepath):
    """Reads unique lemmas from the 'recepient_word' column."""
    lemmas = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'recepient_word' not in reader.fieldnames:
                print(f"Error: Column 'recepient_word' missing in {filepath}")
                sys.exit(1)
            
            for row in reader:
                word = row.get('recepient_word', '').strip().lower()
                if word:
                    lemmas.add(word)
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
        
    return lemmas

def print_stat(label, value, total=None):
    if total:
        percent = (value / total) * 100 if total > 0 else 0
        print(f"{label:30}: {value} ({percent:.1f}%)")
    else:
        print(f"{label:30}: {value}")

def main():
    parser = argparse.ArgumentParser(
        description="Compare lemma overlap between two CSV files."
    )
    parser.add_argument("file_a", help="First CSV file path (e.g., dictionary scan)")
    parser.add_argument("file_b", help="Second CSV file path (e.g., wiktionary dump)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print sample overlapping words")
    args = parser.parse_args()

    print("--- Loading Data ---")
    lemmas_a = get_lemmas_from_csv(args.file_a)
    print(f"File A loaded: {len(lemmas_a)} unique lemmas")
    
    lemmas_b = get_lemmas_from_csv(args.file_b)
    print(f"File B loaded: {len(lemmas_b)} unique lemmas")

    # Set Operations
    intersection = lemmas_a.intersection(lemmas_b)
    only_in_a = lemmas_a.difference(lemmas_b)
    only_in_b = lemmas_b.difference(lemmas_a)
    union = lemmas_a.union(lemmas_b)

    # Metrics
    jaccard_index = len(intersection) / len(union) if len(union) > 0 else 0

    print("\n--- Statistics ---")
    print_stat("Total Unique Words (Union)", len(union))
    print_stat("Overlap (Intersection)", len(intersection))
    print_stat("Jaccard Similarity Index", f"{jaccard_index:.4f}")
    
    print("\n--- Coverage ---")
    print_stat("Unique to File A", len(only_in_a), len(lemmas_a))
    print_stat("Unique to File B", len(only_in_b), len(lemmas_b))
    print_stat("File A covered by File B", len(intersection), len(lemmas_a))
    print_stat("File B covered by File A", len(intersection), len(lemmas_b))

    if args.verbose and intersection:
        print("\n--- Sample Overlap (First 10) ---")
        print(", ".join(sorted(list(intersection))[:10]))

if __name__ == "__main__":
    main()
