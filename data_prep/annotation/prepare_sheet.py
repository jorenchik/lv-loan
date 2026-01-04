import argparse
import csv
import re
import pandas as pd
import sys
import os
from collections import defaultdict
from tqdm import tqdm

from data_prep.corpus.cqp import (
  query_cqp,
  parse_cqp_line,
  score_sentence,
  DEFAULT_CQP_BIN,
  DEFAULT_CQP_DIR,
  DEFAULT_CORPUS,
)
from data_prep.wordnet.wordnet import WordNet
from data_prep.translate.translate import translate_lv_to_en_batch, load_model

def get_tagged_lemmas(parsed_result, lemma_set_lower, primary_lemma):
  """Return dict mapping tag_num -> lemma for all loanwords in sentence."""
  if parsed_result is None:
    return {1: primary_lemma}

  loanword_positions = []
  for i, token in enumerate(parsed_result.tokens):
    token_lemma = token.lemma.lower()
    if token_lemma in lemma_set_lower:
      orig_lemma = lemma_set_lower[token_lemma]
      is_primary = token_lemma == primary_lemma.lower()
      loanword_positions.append((i, orig_lemma, is_primary))

  if not loanword_positions:
    return {1: primary_lemma}

  loanword_positions.sort(key=lambda x: (not x[2], x[0]))
  return {
    tag_num: lemma
    for tag_num, (_, lemma, _) in enumerate(loanword_positions, start=1)
  }


def tag_all_loanwords(parsed_result, lemma_set_lower, primary_lemma):
  """Tag all loanwords in sentence with L1, L2, L3..."""
  if parsed_result is None:
    return None

  loanword_positions = []
  for i, token in enumerate(parsed_result.tokens):
    token_lemma = token.lemma.lower()
    if token_lemma in lemma_set_lower:
      is_primary = token_lemma == primary_lemma.lower()
      loanword_positions.append((i, token.word, is_primary))

  if not loanword_positions:
    return None

  loanword_positions.sort(key=lambda x: (not x[2], x[0]))

  tag_map = {}
  for tag_num, (pos, _, _) in enumerate(loanword_positions, start=1):
    tag_map[pos] = tag_num

  tokens = []
  for i, t in enumerate(parsed_result.tokens):
    if i in tag_map:
      tag_num = tag_map[i]
      tokens.append(f"<L{tag_num}>{t.word}</L{tag_num}>")
    else:
      tokens.append(t.word)

  return " ".join(tokens)


def get_best_sentence_per_lemma(
  word, corpus, cqp_bin, cqp_dir, lemma_set_lower, do_score=True
):
  """Query single lemma, return best sentence and parsed result."""
  search_query = f'[lemma="{word}"]'
  raw_output = query_cqp(corpus, search_query, 50, cqp_bin, cqp_dir)

  results = []
  if raw_output:
    for line in raw_output.split("\n"):
      if not line.strip():
        continue
      parsed = parse_cqp_line(line)
      if parsed:
        results.append(parsed)

  if not results:
    return None, word, None

  if do_score:
    best = max(results, key=score_sentence)
  else:
    best = results[0]

  sentence = tag_all_loanwords(best, lemma_set_lower, word)
  return sentence, word, best


def get_sentences_with_fallback(
  lemmas, corpus, cqp_bin, cqp_dir, lemma_set_lower, do_score=True, limit=100000
):
  """Query all lemmas at once with limit, then fallback for missing ones."""
  escaped = [re.escape(lemma) for lemma in lemmas]
  pattern = "|".join(escaped)
  search_query = f'[lemma="{pattern}"]'

  print(f"Querying {len(lemmas)} lemmas (limit: {limit})...")
  raw_output = query_cqp(corpus, search_query, limit, cqp_bin, cqp_dir)

  best_by_lemma = {}
  lemma_lower_to_orig = {lemma.lower(): lemma for lemma in lemmas}
  total_lines = 0

  if raw_output:
    lines = raw_output.split("\n")
    for line in tqdm(lines, desc="Processing bulk results", unit="line"):
      if not line.strip():
        continue

      total_lines += 1
      parsed = parse_cqp_line(line)
      if not parsed or not parsed.target_word:
        continue

      lemma_lower = parsed.target_word.lemma.lower()
      if lemma_lower not in lemma_lower_to_orig:
        continue

      score = score_sentence(parsed) if do_score else 0
      if lemma_lower not in best_by_lemma or score > best_by_lemma[lemma_lower][0]:
        best_by_lemma[lemma_lower] = (score, parsed)

  found_count = len(best_by_lemma)
  print(
    f"Bulk query: {total_lines} sentences, "
    f"{found_count}/{len(lemmas)} lemmas found"
  )

  missing = [
    lemma_lower_to_orig[ll]
    for ll in lemma_lower_to_orig
    if ll not in best_by_lemma
  ]

  if missing:
    print(f"Fallback: querying {len(missing)} missing lemmas individually...")
    for lemma in tqdm(missing, desc="Fallback queries", unit="word"):
      sentence, _, parsed = get_best_sentence_per_lemma(
        lemma, corpus, cqp_bin, cqp_dir, lemma_set_lower, do_score
      )
      if parsed:
        score = score_sentence(parsed) if do_score else 0
        best_by_lemma[lemma.lower()] = (score, parsed)

  output = {}
  for lemma_lower, lemma_orig in lemma_lower_to_orig.items():
    if lemma_lower not in best_by_lemma:
      output[lemma_orig] = (None, lemma_orig, None)
      continue

    _, best = best_by_lemma[lemma_lower]
    sentence = tag_all_loanwords(best, lemma_set_lower, lemma_orig)
    output[lemma_orig] = (sentence, lemma_orig, best)

  return output


def count_loanwords_in_sentence(parsed_result, lemma_set_lower):
  """Count how many loanwords from the list appear in a parsed sentence."""
  if parsed_result is None:
    return set()

  matched_lemmas = set()
  for token in parsed_result.tokens:
    token_lemma = token.lemma.lower()
    if token_lemma in lemma_set_lower:
      matched_lemmas.add(lemma_set_lower[token_lemma])

  return matched_lemmas


def count_tags_in_sentence(sentence):
  """Count how many <L#> tags are in the sentence."""
  if not sentence:
    return 1
  matches = re.findall(r"<L(\d+)>", sentence)
  return len(matches) if matches else 1


def strip_tags(sentence):
  """Remove <L1></L1> etc. tags for translation."""
  return re.sub(r"</?[LN]\d+>", "", sentence)


def create_native_template(sentence_with_loan_tags):
  """Replace <L1>word</L1> with <N1>word</N1> (keeping the word)."""

  def replace_tag(match):
    tag_num = match.group(1)
    word = match.group(2)
    return f"<N{tag_num}>{word}</N{tag_num}>"

  return re.sub(r"<L(\d+)>([^<]+)</L\d+>", replace_tag, sentence_with_loan_tags)


def format_word_synonyms(word, wn: WordNet):
  """Format synonyms for a single word, grouped by sense."""
  result = wn.get_synonym_groups(word)
  if not result.found:
    return ""

  lines = []
  sense_num = 1
  for entry in result.entries:
    for sense in entry.senses:
      syns = set(sense.synonyms) if sense.synonyms else set()
      syns.discard(word)
      if not syns:
        continue

      definition = getattr(sense, "definition", "") or ""
      syn_str = ", ".join(sorted(syns))

      if definition:
        lines.append(f"{sense_num}. [{definition}]: {syn_str}")
      else:
        lines.append(f"{sense_num}. {syn_str}")
      sense_num += 1

  return "\n".join(lines)


def format_synonyms(tagged_lemmas, wn: WordNet):
  """Format synonyms grouped by loanword tag and sense, always showing word."""
  lines = []
  for tag_num in sorted(tagged_lemmas.keys()):
    word = tagged_lemmas[tag_num]
    word_syns = format_word_synonyms(word, wn)
    if word_syns:
      lines.append(f"L{tag_num} ({word}):")
      for line in word_syns.split("\n"):
        lines.append(f"  {line}")

  return "\n".join(lines)


def format_etymology(tagged_lemmas, lemma_to_lang_info, char_limit=450):
  """Format etymology for each loanword, with character limit per word."""
  lines = []
  for tag_num in sorted(tagged_lemmas.keys()):
    word = tagged_lemmas[tag_num]
    lang_info = lemma_to_lang_info.get(word.lower(), "").strip()
    if lang_info:
      if len(lang_info) > char_limit:
        lang_info = lang_info[: char_limit - 3] + "..."
      lines.append(f"L{tag_num} ({word}): {lang_info}")
    else:
      lines.append(f"L{tag_num} ({word}):")

  return "\n".join(lines)


def format_donor_fields(tagged_lemmas, lemma_to_donor_lang, lemma_to_donor_word):
  """Format donor language and word for each loanword tag."""
  lang_lines = []
  word_lines = []

  for tag_num in sorted(tagged_lemmas.keys()):
    word = tagged_lemmas[tag_num]
    donor_lang = lemma_to_donor_lang.get(word.lower(), "")
    donor_word = lemma_to_donor_word.get(word.lower(), "")
    lang_lines.append(f"{tag_num}- {donor_lang}")
    word_lines.append(f"{tag_num}- {donor_word}")

  return "\n".join(lang_lines), "\n".join(word_lines)


def build_row(
  rec_word,
  sentence_loan,
  row,
  wn,
  parsed_result,
  lemma_set_lower,
  lemma_to_lang_info,
  lemma_to_donor_lang,
  lemma_to_donor_word,
):
  """Build a single output row dict."""
  if sentence_loan is None:
    sentence_loan = f"<L1>{rec_word}</L1>"
    found = False
  else:
    found = True

  sentence_native = create_native_template(sentence_loan)
  tagged_lemmas = get_tagged_lemmas(parsed_result, lemma_set_lower, rec_word)
  synonyms = format_synonyms(tagged_lemmas, wn)
  etymology = format_etymology(tagged_lemmas, lemma_to_lang_info)
  donor_lang, donor_word = format_donor_fields(
    tagged_lemmas, lemma_to_donor_lang, lemma_to_donor_word
  )

  return {
    "Loanword sentence": sentence_loan,
    "Native sentence": sentence_native,
    "Target": "",
    "Valid.": False,
    "Donor lang.": donor_lang,
    "Donor word": donor_word,
    "Suggestions": synonyms,
    "Etymology": etymology,
  }, found


def print_results_summary(stats):
  """Print detailed results summary."""
  total = stats["total"]
  found = stats["found"]
  not_found = stats["not_found"]

  print("\n" + "=" * 60)
  print("RESULTS SUMMARY")
  print("=" * 60)
  print(f"Total lemmas processed: {total}")
  print(f"  Found in corpus:      {found} ({100*found/total:.1f}%)")
  print(f"  Not found:            {not_found} ({100*not_found/total:.1f}%)")

  if stats["by_donor"]:
    print("\nBy donor language:")
    print("-" * 40)
    for lang in sorted(stats["by_donor"].keys()):
      lang_stats = stats["by_donor"][lang]
      lang_total = lang_stats["found"] + lang_stats["not_found"]
      print(
        f"  {lang or '(unknown)':<20} "
        f"{lang_stats['found']:>4}/{lang_total:<4} "
        f"({100*lang_stats['found']/lang_total:.1f}%)"
      )

  if stats["sentence_matches"]:
    print("\nSentences by loanword count:")
    print("-" * 40)
    match_counts = defaultdict(int)

    for sentence_text, matched_lemmas in stats["sentence_matches"].items():
      count = len(matched_lemmas)
      match_counts[count] += 1

    for count in sorted(match_counts.keys(), reverse=True):
      num_sentences = match_counts[count]
      label = "loanword" if count == 1 else "loanwords"
      print(f"  {count} {label}: {num_sentences} sentences")

  if stats["not_found_lemmas"]:
    print(f"\nNot found lemmas ({len(stats['not_found_lemmas'])})")

  print("=" * 60 + "\n")


def process_entries(
  input_rows,
  args,
  lemma_set_lower,
  lemma_to_lang_info,
  lemma_to_donor_lang,
  lemma_to_donor_word,
  wn,
):
  """Process all entries using the selected strategy."""
  found_rows = []
  not_found_rows = []
  found_sentences = []
  not_found_sentences = []

  stats = {
    "total": 0,
    "found": 0,
    "not_found": 0,
    "by_donor": defaultdict(lambda: {"found": 0, "not_found": 0}),
    "not_found_lemmas": [],
    "sentence_matches": {},
  }

  if args.strategy == "streaming":
    lemmas = [
      row.get("recepient_word", "").strip()
      for row in input_rows
      if row.get("recepient_word", "").strip()
    ]

    sentence_map = get_sentences_with_fallback(
      lemmas,
      args.corpus,
      args.cqp_bin,
      args.cqp_dir,
      lemma_set_lower,
      do_score=args.score,
      limit=args.query_limit,
    )

    for row in tqdm(input_rows, desc="Building rows", unit="word"):
      rec_word = row.get("recepient_word", "").strip()
      if not rec_word:
        continue

      sentence_loan, _, parsed_result = sentence_map.get(
        rec_word, (None, rec_word, None)
      )
      output_row, found = build_row(
        rec_word,
        sentence_loan,
        row,
        wn,
        parsed_result,
        lemma_set_lower,
        lemma_to_lang_info,
        lemma_to_donor_lang,
        lemma_to_donor_word,
      )
      sentence_text = strip_tags(output_row["Loanword sentence"])

      stats["total"] += 1
      donor_lang = row.get("donor_language", "")

      if found:
        stats["found"] += 1
        stats["by_donor"][donor_lang]["found"] += 1
        found_rows.append(output_row)
        found_sentences.append(sentence_text)

        matched_in_sentence = count_loanwords_in_sentence(
          parsed_result, lemma_set_lower
        )
        if sentence_text not in stats["sentence_matches"]:
          stats["sentence_matches"][sentence_text] = matched_in_sentence
        else:
          stats["sentence_matches"][sentence_text].update(matched_in_sentence)
      else:
        stats["not_found"] += 1
        stats["by_donor"][donor_lang]["not_found"] += 1
        stats["not_found_lemmas"].append(rec_word)
        not_found_rows.append(output_row)
        not_found_sentences.append(sentence_text)

  else:  # per_lemma
    for row in tqdm(input_rows, desc="Extracting", unit="word"):
      rec_word = row.get("recepient_word", "").strip()
      if not rec_word:
        continue

      sentence_loan, _, parsed_result = get_best_sentence_per_lemma(
        rec_word,
        args.corpus,
        args.cqp_bin,
        args.cqp_dir,
        lemma_set_lower,
        do_score=args.score,
      )
      output_row, found = build_row(
        rec_word,
        sentence_loan,
        row,
        wn,
        parsed_result,
        lemma_set_lower,
        lemma_to_lang_info,
        lemma_to_donor_lang,
        lemma_to_donor_word,
      )
      sentence_text = strip_tags(output_row["Loanword sentence"])

      stats["total"] += 1
      donor_lang = row.get("donor_language", "")

      if found:
        stats["found"] += 1
        stats["by_donor"][donor_lang]["found"] += 1
        found_rows.append(output_row)
        found_sentences.append(sentence_text)

        matched_in_sentence = count_loanwords_in_sentence(
          parsed_result, lemma_set_lower
        )
        if sentence_text not in stats["sentence_matches"]:
          stats["sentence_matches"][sentence_text] = matched_in_sentence
        else:
          stats["sentence_matches"][sentence_text].update(matched_in_sentence)
      else:
        stats["not_found"] += 1
        stats["by_donor"][donor_lang]["not_found"] += 1
        stats["not_found_lemmas"].append(rec_word)
        not_found_rows.append(output_row)
        not_found_sentences.append(sentence_text)

  return found_rows, not_found_rows, found_sentences, not_found_sentences, stats


def write_output(args, rows_to_write, column_widths_cm):
  """Write output to CSV or Excel."""
  df = pd.DataFrame(rows_to_write)

  if args.output.endswith(".csv"):
    df.to_csv(args.output, index=False)
    print(f"Done. Saved to {args.output}")
    return

  writer = pd.ExcelWriter(args.output, engine="xlsxwriter")
  df.to_excel(writer, index=False, sheet_name="Sheet1")

  workbook = writer.book
  worksheet = writer.sheets["Sheet1"]
  wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

  for i, w_cm in enumerate(column_widths_cm):
    excel_width = w_cm * 3.89
    worksheet.set_column(i, i, excel_width, wrap_format)

  writer.close()


def main():
  parser = argparse.ArgumentParser(
    description="Generate ConLoan-compliant annotation spreadsheet."
  )
  parser.add_argument(
    "inputs", nargs="+", help="One or more input CSV files to concatenate"
  )
  parser.add_argument(
    "--wordnet-xml", required=True, help="Path to WordNet LMF XML"
  )
  parser.add_argument(
    "--output", default="conloan_annotation.xlsx", help="Output Excel file"
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Translation batch size"
  )
  parser.add_argument(
    "--translate",
    action="store_true",
    default=False,
    help="Enable translation (default: False)",
  )
  parser.add_argument(
    "--score",
    action="store_true",
    default=False,
    help="Enable sentence scoring for best selection (default: False)",
  )
  parser.add_argument(
    "--strategy",
    choices=["per_lemma", "streaming"],
    default="streaming",
    help="Query strategy: per_lemma or streaming (default: streaming)",
  )
  parser.add_argument(
    "--query-limit",
    type=int,
    default=100000,
    help="Max results for bulk query before fallback (default: 100000)",
  )
  parser.add_argument("--corpus", default=DEFAULT_CORPUS)
  parser.add_argument("--cqp-bin", default=DEFAULT_CQP_BIN)
  parser.add_argument("--cqp-dir", default=DEFAULT_CQP_DIR)
  args = parser.parse_args()

  if not os.path.exists(args.wordnet_xml):
    print(f"Error: WordNet file {args.wordnet_xml} not found.", file=sys.stderr)
    sys.exit(1)

  print("Initializing WordNet...")
  wn = WordNet(args.wordnet_xml)

  if args.translate:
    print("Loading translation model...")
    load_model()

  input_rows = []
  for file_path in args.inputs:
    if not os.path.exists(file_path):
      print(f"Warning: File {file_path} not found. Skipping.", file=sys.stderr)
      continue
    with open(file_path, mode="r", encoding="utf-8") as f:
      input_rows.extend(list(csv.DictReader(f)))

  if not input_rows:
    print("Error: No data found in provided input files.", file=sys.stderr)
    sys.exit(1)

  all_lemmas = [
    row.get("recepient_word", "").strip()
    for row in input_rows
    if row.get("recepient_word", "").strip()
  ]
  lemma_set_lower = {lemma.lower(): lemma for lemma in all_lemmas}

  lemma_to_lang_info = {
    row.get("recepient_word", "").strip().lower(): row.get("lang_info", "").strip()
    for row in input_rows
    if row.get("recepient_word", "").strip()
  }

  lemma_to_donor_lang = {
    row.get("recepient_word", "").strip().lower(): row.get(
      "donor_language", ""
    ).strip()
    for row in input_rows
    if row.get("recepient_word", "").strip()
  }

  lemma_to_donor_word = {
    row.get("recepient_word", "").strip().lower(): row.get("donor_word", "").strip()
    for row in input_rows
    if row.get("recepient_word", "").strip()
  }

  print(f"Processing {len(input_rows)} entries (strategy: {args.strategy})...")

  found_rows, not_found_rows, found_sentences, not_found_sentences, stats = (
    process_entries(
      input_rows,
      args,
      lemma_set_lower,
      lemma_to_lang_info,
      lemma_to_donor_lang,
      lemma_to_donor_word,
      wn,
    )
  )

  rows_to_write = found_rows + not_found_rows
  sentences_to_translate = found_sentences + not_found_sentences
  print_results_summary(stats)

  column_widths_cm = [6.4, 6.4, 6.4, 1.8, 3.0, 3.0, 6.4, 10.0]

  if args.translate:
    print(f"Translating {len(sentences_to_translate)} sentences...")
    translations = []
    for i in tqdm(
      range(0, len(sentences_to_translate), args.batch_size),
      desc="Translating",
      unit="batch",
    ):
      batch = sentences_to_translate[i : i + args.batch_size]
      translations.extend(translate_lv_to_en_batch(batch))

    for row, translation in zip(rows_to_write, translations):
      row["Target"] = translation

  write_output(args, rows_to_write, column_widths_cm)

  print(f"ConLoan annotation file generated: {args.output}")
  print(f"  - {len(found_rows)} matched lemmas")
  print(f"  - {len(not_found_rows)} not found (placeholders appended)")


if __name__ == "__main__":
  main()
