import argparse
import json
import re
import pandas as pd
import sys
import os


def extract_tag_contents(sentence, tag_prefix):
  """Extract tag number -> content mapping from sentence.

  e.g., "<L1>word</L1>" -> {"1": "word"}
  """
  if not sentence:
    return {}

  pattern = rf"<{tag_prefix}(\d+)>([^<]*)</{tag_prefix}\d+>"
  matches = re.findall(pattern, str(sentence))
  return {num: content for num, content in matches}


def strip_all_tags(sentence):
  """Remove all <L#>, </L#>, <N#>, </N#> tags but keep content."""
  if not sentence:
    return ""
  return re.sub(r"</?[LN]\d+>", "", str(sentence))


def fill_blank_n_tags(native_sentence, l_words):
  """Replace empty N tags with corresponding L word content."""
  result = native_sentence
  for tag_num, l_word in l_words.items():
    result = re.sub(
      rf"<N{tag_num}></N{tag_num}>",
      f"<N{tag_num}>{l_word}</N{tag_num}>",
      result,
    )
  return result


def get_actionable_tags(l_words, n_words_raw):
  """Return list of tag numbers that should be kept (replaced or blanked).

  - Retained (L == N, N not blank): filter out
  - Replaced (L != N): keep
  - Blanked (N is empty): keep as self-replacement
  """
  actionable = []
  for tag_num in sorted(l_words.keys(), key=int):
    l_word = l_words.get(tag_num, "")
    n_word = n_words_raw.get(tag_num, "")

    if n_word == "":
      # Blanked → self-replacement
      actionable.append(tag_num)
    elif l_word != n_word:
      # Replaced
      actionable.append(tag_num)
    # Retained (L == N, N not blank) → filter out

  return actionable


def rebuild_sentence_with_filtered_tags(sentence, tag_prefix, actionable_tags):
  """Rebuild sentence keeping only actionable tags, renumbered consecutively.

  Non-actionable tags are stripped but content is kept.
  """
  if not sentence:
    return ""

  result = str(sentence)
  tag_mapping = {old: str(new) for new, old in enumerate(actionable_tags, start=1)}

  # First, replace actionable tags with new numbers
  for old_num, new_num in tag_mapping.items():
    result = re.sub(
      rf"<{tag_prefix}{old_num}>",
      f"<{tag_prefix}{new_num}>",
      result,
    )
    result = re.sub(
      rf"</{tag_prefix}{old_num}>",
      f"</{tag_prefix}{new_num}>",
      result,
    )

  # Then, strip non-actionable tags (keep content)
  all_tag_nums = set(re.findall(rf"<{tag_prefix}(\d+)>", result))
  new_tag_nums = set(tag_mapping.values())
  non_actionable_nums = all_tag_nums - new_tag_nums

  for num in non_actionable_nums:
    result = re.sub(rf"<{tag_prefix}{num}>([^<]*)</{tag_prefix}{num}>", r"\1", result)

  return result


def normalize_target(value):
  """Return None if target is empty/nan, otherwise return string."""
  if pd.isna(value):
    return None
  str_val = str(value).strip()
  if str_val.lower() in ("", "nan", "none"):
    return None
  return str_val


def build_entry(row):
  """Build a JSON entry from a row."""
  loanword_sentence = str(row.get("Loanword sentence", "") or "")
  native_sentence = str(row.get("Native sentence", "") or "")
  target = normalize_target(row.get("Target"))

  l_words = extract_tag_contents(loanword_sentence, "L")
  n_words_raw = extract_tag_contents(native_sentence, "N")

  if not l_words:
    return None

  # Determine which tags to keep (replaced or blanked)
  actionable_tags = get_actionable_tags(l_words, n_words_raw)

  if not actionable_tags:
    return None

  # Fill blank N tags with L content for output
  native_sentence_filled = fill_blank_n_tags(native_sentence, l_words)
  n_words = extract_tag_contents(native_sentence_filled, "N")

  new_loanword = rebuild_sentence_with_filtered_tags(
    loanword_sentence, "L", actionable_tags
  )
  new_native = rebuild_sentence_with_filtered_tags(
    native_sentence_filled, "N", actionable_tags
  )

  new_l_words = {}
  new_n_words = {}
  corresponding_words = {}

  for new_num, old_num in enumerate(actionable_tags, start=1):
    new_key = str(new_num)
    l_word = l_words.get(old_num, "")
    n_word = n_words.get(old_num, "")

    new_l_words[new_key] = l_word
    new_n_words[new_key] = n_word
    corresponding_words[new_key] = [l_word, n_word]

  return {
    "source_annotated_loanwords": new_loanword,
    "source_annotated_loanwords_replaced": new_native,
    "target": target,
    "source_plain": strip_all_tags(loanword_sentence),
    "source_annotated_plain": strip_all_tags(new_native),
    "words_in_L_tags": new_l_words,
    "words_in_N_tags": new_n_words,
    "corresponding_words": corresponding_words,
  }


def main():
  parser = argparse.ArgumentParser(
    description="Extract valid replacements from ConLoan spreadsheet to JSON."
  )
  parser.add_argument("input", help="Input Excel (.xlsx) or CSV file")
  parser.add_argument(
    "--output",
    help="Output JSON file (default: input_replacements.json)",
  )
  parser.add_argument(
    "--valid-col",
    default="Valid.",
    help="Column name for validity flag (default: 'Valid.')",
  )
  args = parser.parse_args()

  if not os.path.exists(args.input):
    print(f"Error: File {args.input} not found.", file=sys.stderr)
    sys.exit(1)

  if args.output:
    output_path = args.output
  else:
    base, _ = os.path.splitext(args.input)
    output_path = f"{base}_replacements.json"

  print(f"Loading {args.input}...")
  if args.input.endswith(".csv"):
    df = pd.read_csv(args.input)
  else:
    df = pd.read_excel(args.input, engine="openpyxl")

  if args.valid_col not in df.columns:
    print(f"Error: Column '{args.valid_col}' not found.", file=sys.stderr)
    print(f"Available columns: {list(df.columns)}", file=sys.stderr)
    sys.exit(1)

  valid_mask = (df[args.valid_col] == True) | (
    df[args.valid_col].astype(str).str.upper() == "TRUE"
  )
  valid_df = df[valid_mask]
  print(f"Found {len(valid_df)} valid entries.")

  entries = []
  skipped_no_replacement = 0
  skipped_no_tags = 0

  for _, row in valid_df.iterrows():
    entry = build_entry(row)
    if entry:
      entries.append(entry)
    else:
      loanword = str(row.get("Loanword sentence", "") or "")
      if not extract_tag_contents(loanword, "L"):
        skipped_no_tags += 1
      else:
        skipped_no_replacement += 1

  print(f"Extracted {len(entries)} entries with replacements.")
  if skipped_no_tags:
    print(f"Skipped {skipped_no_tags} entries (no L tags).")
  if skipped_no_replacement:
    print(f"Skipped {skipped_no_replacement} entries (no changes).")

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(entries, f, ensure_ascii=False, indent=2)

  print(f"Saved to {output_path}")


if __name__ == "__main__":
  main()
