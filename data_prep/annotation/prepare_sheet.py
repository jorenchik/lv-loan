import argparse
import csv
import re
import pandas as pd
import sys
import os
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


def get_best_sentence(word, corpus, cqp_bin, cqp_dir):
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
    return None, word

  scored = sorted(
    [(score_sentence(r), r) for r in results], key=lambda x: x[0], reverse=True
  )
  best_res = scored[0][1]

  tokens = []
  matched_form = word
  for i, t in enumerate(best_res.tokens):
    if i == best_res.match_index:
      matched_form = t.word
      tokens.append(f"<L1>{t.word}</L1>")
    else:
      tokens.append(t.word)
  return " ".join(tokens), matched_form


def strip_tags(sentence):
  """Remove <L1></L1> tags for translation."""
  return re.sub(r"</?[LN]\d+>", "", sentence)


def create_native_template(sentence_with_loan_tag):
  """Replace <L1>word</L1> with <N1></N1> for annotator to fill."""
  return re.sub(r"<L1>[^<]+</L1>", "<N1></N1>", sentence_with_loan_tag)


def format_synonyms(word, wn: WordNet):
  result = wn.get_synonym_groups(word)
  if not result.found:
    return ""

  synonyms = set()
  for entry in result.entries:
    for sense in entry.senses:
      if sense.synonyms:
        synonyms.update(sense.synonyms)

  synonyms.discard(word)
  return "; ".join(sorted(synonyms))


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
  parser.add_argument("--corpus", default=DEFAULT_CORPUS)
  parser.add_argument("--cqp-bin", default=DEFAULT_CQP_BIN)
  parser.add_argument("--cqp-dir", default=DEFAULT_CQP_DIR)
  args = parser.parse_args()

  if not os.path.exists(args.wordnet_xml):
    print(f"Error: WordNet file {args.wordnet_xml} not found.", file=sys.stderr)
    sys.exit(1)

  print("Initializing WordNet...")
  wn = WordNet(args.wordnet_xml)

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

  rows_to_write = []
  sentences_to_translate = []

  print(f"Processing {len(input_rows)} entries...")
  for row in tqdm(input_rows, desc="Extracting", unit="word"):
    rec_word = row.get("recepient_word", "").strip()
    if not rec_word:
      continue

    sentence_loan, _ = get_best_sentence(
      rec_word, args.corpus, args.cqp_bin, args.cqp_dir
    )

    if sentence_loan is None:
      sentence_loan = f"<L1>{rec_word}</L1>"

    sentence_native = create_native_template(sentence_loan)
    synonyms = format_synonyms(rec_word, wn)

    rows_to_write.append(
      {
        "Loanword sentence": sentence_loan,
        "Native sentence": sentence_native,
        "Target": "",
        "Valid.": False,
        "Donor lang.": row.get("donor_language", ""),
        "Donor word": row.get("donor_word", ""),
        "Suggestions": synonyms,
        "Etymology": row.get("lang_info", "").strip(),
      }
    )
    sentences_to_translate.append(strip_tags(sentence_loan))

  column_widths_cm = [6.4, 6.4, 6.4, 1.8, 3.0, 3.0, 6.4, 10.0]

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
  print(f"ConLoan annotation file generated: {args.output}")
  print(f"  - {len(rows_to_write)} loanword instances")


if __name__ == "__main__":
  main()
