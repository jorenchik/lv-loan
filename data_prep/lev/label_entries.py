import argparse
import csv
import re
import sys

# --- CONFIGURATION ---

RECIPIENT_LANG = 'lv'

# Indicators of a loanword in the text
LOAN_INDICATORS = [
    r"\baizguvums\b", 
    r"\baizg\.", 
    r"\baizgūts\b",
    r"\baizg\b"
]

# Context window size (WORDS) around found markers
CONTEXT_SIZE = 5

# Dictionary cues/abbreviations to mark (NOT languages)
CUE_PATTERNS = {
    "aizg.", "etim.", "apv.", "sal.", "sk.", "var.",
    "liter.", "dial.", "arh.", "vēst.", "atv.",
}

# Primary known languages/families to validate candidates
KNOWN_LANGUAGES = {
    # Abbreviations
    'v.': 'vācu', 'vlv.': 'viduslejasvācu', 'vh.': 'vidusaugšvācu',
    'kr.': 'krievu', 'skr.': 'senkrievu', 'bkr.': 'baltkrievu',
    'līb.': 'lībiešu', 'ig.': 'igauņu', 'somu': 'somu',
    'lat.': 'latīņu', 'gr.': 'grieķu', 'fr.': 'franču',
    'it.': 'itāļu', 'ang.': 'angļu', 'a.': 'angļu',
    'zv.': 'zviedru', 'sl.': 'slāvu', 'psl.': 'prāslāvu',
    'ukr.': 'ukraiņu', 'bulg.': 'bulgāru', 'č.': 'čehu',
    'p.': 'poļu', 'go.': 'gotu', 'pr.': 'prūšu',
    'lš.': 'lietuviešu', 'liet.': 'lietuviešu',
    'ssk.': 'sanskrita', 'ide.': 'indoeiropiešu',
    'b.': 'baltu', 'ab.': 'austrumbaltu',
    
    # Full names / Genitive forms
    'vācu': 'vācu', 'krievu': 'krievu', 'latīņu': 'latīņu',
    'grieķu': 'grieķu', 'franču': 'franču', 'angļu': 'angļu',
    'zviedru': 'zviedru', 'igauņu': 'igauņu', 'lībiešu': 'lībiešu',
    'lietuviešu': 'lietuviešu', 'slāvu': 'slāvu', 'ģermāņu': 'ģermāņu',
    'semītu': 'semītu', 'tirku': 'tirku', 'somu': 'somu',
    'baltu': 'baltu', 'irāņu': 'irāņu', 'skandināvu': 'skandināvu',
    'viduslejasvācu': 'viduslejasvācu',
    
    # Additional from PDF
    'alb.': 'albaņu', 'arm.': 'armēņu', 'asor.': 'augšsorbu',
    'bsl.': 'baznīcslāvu', 'b-sl.': 'baltu-slāvu', 'bv.': 'baltvācu',
    'd.': 'dāņu', 'dor.': 'doriešu', 'he.': 'hetu',
    'isļ.': 'islandiešu', 'jlat.': 'jaunlatīņu', 'narev.': 'Narevas baltu',
    'norv.': 'norvēģu', 'oset.': 'osetīnu', 'rir.': 'rietumirāņu',
    'rum.': 'rumāņu', 'sa.': 'senangļu', 'sč.': 'senčehu',
    'sebr.': 'senebreju', 'sfr.': 'senfranču', 'sfrī.': 'senfrīzu',
    's-h.': 'serbhorvātu', 'si.': 'senindiešu', 'sisl.': 'senislandiešu',
    'sp.': 'senpoju', 'spers.': 'senpersu', 'ssak.': 'sensakšu',
    'ssl.': 'senslāvu', 's-u.': 'somugru', 'szv.': 'senzviedru',
    'tadž.': 'tadžiku', 'toh.': 'tohāru', 'trāķ.': 'trāķiešu',
    'ung.': 'ungāru', 'vav.': 'vidusaugšvācu', 'vlat.': 'viduslatīņu',
    'vv.': 'vidusvācu', 'la.': 'latviešu', 'kurs.': 'kursisms',
    'kursen.': 'kursenieku'
}

def clean_text(text):
    """Normalizes whitespace."""
    if not text:
        return ""
    return " ".join(str(text).split())

def is_loanword_candidate(text):
    """Fast check if regex indicators exist."""
    text_lower = text.lower()
    for key in LOAN_INDICATORS:
        if re.search(key, text_lower):
            return True
    return False

def extract_language_contexts(text):
    """
    Finds all known language markers in text and extracts WORD-LEVEL context windows.
    Returns list of context strings and list of found language keys.
    """
    contexts = []
    seen_positions = set()
    found_langs = set()
    
    # Sort by length descending to match longer patterns first
    pattern_keys = sorted(KNOWN_LANGUAGES.keys(), key=len, reverse=True)
    
    for pattern_key in pattern_keys:
        escaped_key = re.escape(pattern_key)
        
        if pattern_key.endswith('.'):
            pattern = r"(?<!\w)" + escaped_key
        else:
            pattern = r"\b" + escaped_key + r"\b"
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start, end = match.start(), match.end()
            
            # Skip duplicates
            if any(abs(start - s) < 3 for s in seen_positions):
                continue
            seen_positions.add(start)
            
            # Use ORIGINAL matched text (preserves case and parentheses)
            original_match = match.group(0)
            found_langs.add(pattern_key)
            
            # Split text into words for context
            words = text.split()
            word_indices = []
            
            # Build word index mapping
            pos = 0
            for i, word in enumerate(words):
                word_indices.append((pos, pos + len(word)))
                pos += len(word) + 1
            
            # Find which word index contains our match
            word_idx = -1
            for i, (w_start, w_end) in enumerate(word_indices):
                if w_start <= start < w_end:
                    word_idx = i
                    break
            
            if word_idx == -1:
                continue
            
            # Get context words range
            left = max(0, word_idx - CONTEXT_SIZE)
            right = min(len(words), word_idx + CONTEXT_SIZE + 1)
            
            # Extract context words with language tag
            context_words = words[left:word_idx] + ['<L>' + original_match + '</L>'] + words[word_idx+1:right]
            
            # Clean newlines
            cleaned = [w.replace('\n', ' ').replace('\r', '') for w in context_words]
            
            contexts.append(" ".join(cleaned))
    
    return contexts, list(found_langs)

def mark_cues_in_context(context_str, cue_patterns):
    """
    Within a language context string, mark any dictionary cues found.
    Returns the marked string.
    """
    marked = context_str
    
    # Sort cues by length descending to match longer patterns first
    sorted_cues = sorted(cue_patterns, key=len, reverse=True)
    
    for cue in sorted_cues:
        if not cue:
            continue
        escaped_cue = re.escape(cue)
        
        if cue.endswith('.'):
            pattern = r"(?<!\w)" + escaped_cue + r"(?!\w)"
        else:
            pattern = r"\b" + escaped_cue + r"\b"
        
        # Only replace if not already inside a tag
        def replace_if_not_tagged(match):
            matched_text = match.group(0)
            # Check if already tagged
            if '<C>' in matched_text or '</C>' in matched_text:
                return matched_text
            return '<C>' + matched_text + '</C>'
        
        marked = re.sub(pattern, replace_if_not_tagged, marked, flags=re.IGNORECASE)
    
    return marked

def main():
    parser = argparse.ArgumentParser(description="Extract loanwords with language and cue context.")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("output", help="Output CSV file path")
    args = parser.parse_args()

    results = []
    rows_processed = 0

    print(f"Reading {args.input}...")

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            text_col = None
            headword_col = None
            
            if reader.fieldnames:
                for col in reader.fieldnames:
                    if 'text' in col.lower(): text_col = col
                    if 'headword' in col.lower() or 'word' in col.lower(): headword_col = col
            
            if not text_col or not headword_col:
                f.seek(0)
                reader = csv.reader(f)
                has_header = False 
                idx_word = 1 
                idx_text = 3
            else:
                has_header = True

            for row in reader:
                rows_processed += 1
                
                if has_header:
                    headword = row[headword_col]
                    raw_text = row[text_col]
                else:
                    if not row or len(row) <= idx_text: continue
                    headword = row[idx_word]
                    raw_text = row[idx_text]

                text = clean_text(raw_text)

                if is_loanword_candidate(text):
                    # Step 1: Extract language contexts
                    lang_contexts, lang_keys = extract_language_contexts(text)
                    
                    # Step 2: Mark cues within each language context
                    marked_contexts = []
                    for ctx in lang_contexts:
                        marked_ctx = mark_cues_in_context(ctx, CUE_PATTERNS)
                        marked_contexts.append(marked_ctx)
                    
                    lang_info = "\n".join(marked_contexts) if marked_contexts else ""
                    
                    results.append({
                        'recepient_word': headword,
                        'donor_language': "",
                        'donor_word': "",
                        'lang_info': lang_info
                    })

    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    print(f"Scanned {rows_processed} rows.")
    print(f"Found {len(results)} potential loanwords.")

    # Write output
    fieldnames = ['recepient_word', 'donor_language', 'donor_word', 'lang_info']
    try:
        with open(args.output, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved to {args.output}")
        
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    main()
