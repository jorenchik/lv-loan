import csv
import argparse
import re
from collections import Counter

# Mapping of Wiktionary template codes to human-readable text
TEMPLATE_READABLE = {
  "bor": "borrowed from",
  "lbor": "learned borrowing from",
  "mbor": "possibly borrowed from",
  "der": "derived from",
  "cog": "cognate with",
  "aff": "affix from",
  "suff": "suffix from",
  "pref": "prefix from",
  "inch": "inherited from",
  "relbor": "related borrowing from",
}

# Language code to full name mapping
LANGUAGE_NAMES = {
  "en": "English", "de": "German", "fr": "French", "it": "Italian",
  "es": "Spanish", "pt": "Portuguese", "la": "Latin", "grc": "Ancient Greek",
  "sv": "Swedish", "da": "Danish", "no": "Norwegian", "is": "Icelandic",
  "fo": "Faroese", "nl": "Dutch", "mhn": "Middle High German",
  "lv": "Latvian", "lt": "Lithuanian", "pl": "Polish", "ru": "Russian",
  "cs": "Czech", "sk": "Slovak", "bg": "Bulgarian", "uk": "Ukrainian",
  "non": "Old Norse", "gem-pro": "Proto-Germanic", "ine-pro": "Proto-Indo-European",
  "sa": "Sanskrit", "hi": "Hindi", "fa": "Persian", "ar": "Arabic",
  "he": "Hebrew", "et": "Estonian", "fi": "Finnish", "hu": "Hungarian",
  "ga": "Irish", "cy": "Welsh", "br": "Breton", "sga": "Old Irish",
  "got": "Gotlandic", "goh": "Old High German", "osx": "Old Saxon",
  "ang": "Old English", "ofm": "Middle French", "fro": "Old French",
  # Add more as needed
}

def extract_templates(text):
  """Extract all {{...}} templates and split by pipe."""
  pattern = r"\{\{([^}]+)\}\}"
  matches = re.findall(pattern, text)
  return [[elem.strip() for elem in match.split("|")] for match in matches]

def format_template(template):
  """Convert a template list into marked-up string."""
  if not template:
    return None
  
  t_type = template[0].lower()
  
  # Skip non-relevant templates
  if t_type in {"rfe", "suffix", "af", "inh", "cat", "考证", "zh-pron", "w", "cog"}:
    return None
  
  readable_prefix = TEMPLATE_READABLE.get(t_type)
  if not readable_prefix:
    return None
  
  # Handle template structures
  if t_type in {"bor", "lbor", "mbor", "der", "aff", "suff", "pref", "inch", "relbor"}:
    # {{type|lv|src_lang|word}}
    if len(template) >= 3:
      lang_code = template[2] if len(template) > 3 else template[2]
      word = template[3] if len(template) > 3 else ""
      
      lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
      
      return f"<C>{readable_prefix}</C> <L>{lang_name}</L> {word}".strip()
  
  elif t_type == "cog":
    # {{cog|lang|word}}
    if len(template) >= 3:
      lang_code = template[1]
      word = template[2] if len(template) > 2 else ""
      
      lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
      
      return f"<C>{readable_prefix}</C> <L>{lang_name}</L> {word}".strip()
  
  return None

def read_csv(csv_in, csv_out):
  counter = Counter()
  results = []
  word_set = set()

  with open(csv_in, "r", encoding="utf-8", errors="ignore") as in_:
    reader = csv.DictReader(in_)
    
    for row in reader:
      word = row["word"]
      lang = row["language"].lower().strip()
      etymology = row["etymology_text"]
      
      if lang != "latvian":
        continue
      
      if word and word[0].isupper():
        counter.update(["skipped_uppercase"])
        continue

      if word and len(word) <= 2:
        counter.update(["skipped_too_short"])
        continue

      if word and (word[0] == '-' or word[-1] == '-'):
        counter.update(["skipped_prefix_suffix"])
        continue
      
      templates = extract_templates(etymology)
      if not templates:
        counter.update(["skipped_no_templates"])
        continue
      
      relevant_templates = []
      for t in templates:
        t_type = t[0].lower()
        if t_type in TEMPLATE_READABLE:
          if len(t) > 1 and t[1] != "lv":
            continue
          if len(t) > 2 and "-pro" in t[2].lower():
            continue
          relevant_templates.append(t)
      
      if not relevant_templates:
        counter.update(["skipped_no_relevant"])
        continue
      
      counter.update(["processed"])
      
      readable_entries = []
      for t in relevant_templates:
        formatted = format_template(t)
        if formatted:
          readable_entries.append(formatted)
      
      if not readable_entries:
        continue
      
      # Join with newlines
      lang_info = "\n".join(readable_entries)
      
      if word not in word_set:
        word_set.add(word)
      
      results.append({
        "recepient_word": word,
        "donor_language": "",
        "donor_word": "",
        "lang_info": lang_info
      })

  with open(csv_out, "w", encoding="utf-8", newline="") as out:
    fieldnames = ["recepient_word", "donor_language", "donor_word", "lang_info"]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
  
  print("\n=== Summary ===")
  for item, count in counter.most_common():
    print(f"{item:25s}: {count}")
  print(f"\nTotal unique words: {len(word_set)}")
  print(f"Total rows: {len(results)}")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("csv_in", help="Input CSV file path")
  parser.add_argument("csv_out", help="Output CSV file path")
  args = parser.parse_args()
  
  read_csv(args.csv_in, args.csv_out)

if __name__ == "__main__":
  main()
