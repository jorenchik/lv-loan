import argparse
import re
import pandas as pd
import sys
import os
from tqdm import tqdm

from data_prep.translate.translate import translate_lv_to_en_batch, load_model

def strip_tags(sentence):
  """Remove <L1></L1> etc. tags for translation."""
  return re.sub(r"</?[LN]\d+>", "", sentence)

def main():
  parser = argparse.ArgumentParser(
    description="Translate Loanword sentences in existing spreadsheet."
  )
  parser.add_argument("input", help="Input Excel (.xlsx) or CSV file")
  parser.add_argument(
    "--output",
    help="Output file (default: overwrites input or adds _translated suffix)",
  )
  parser.add_argument(
    "--batch-size", type=int, default=32, help="Translation batch size"
  )
  parser.add_argument(
    "--source-col",
    default="Loanword sentence",
    help="Column to translate (default: 'Loanword sentence')",
  )
  parser.add_argument(
    "--target-col",
    default="Target",
    help="Column to write translations (default: 'Target')",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing translations in target column",
  )
  parser.add_argument(
    "--strip-tags",
    action="store_true",
    default=True,
    help="Strip <L#> and <N#> tags before translation (default: True)",
  )
  parser.add_argument(
    "--no-strip-tags",
    action="store_false",
    dest="strip_tags",
    help="Keep tags in text for translation",
  )
  args = parser.parse_args()

  if not os.path.exists(args.input):
    print(f"Error: File {args.input} not found.", file=sys.stderr)
    sys.exit(1)

  # Determine output path
  if args.output:
    output_path = args.output
  else:
    base, ext = os.path.splitext(args.input)
    output_path = f"{base}_translated{ext}"

  # Load input
  print(f"Loading {args.input}...")
  if args.input.endswith(".csv"):
    df = pd.read_csv(args.input)
  else:
    df = pd.read_excel(args.input)

  if args.source_col not in df.columns:
    print(f"Error: Column '{args.source_col}' not found.", file=sys.stderr)
    print(f"Available columns: {list(df.columns)}", file=sys.stderr)
    sys.exit(1)

  # Ensure target column exists
  if args.target_col not in df.columns:
    df[args.target_col] = ""

  # Identify rows to translate
  if args.overwrite:
    indices = df.index.tolist()
  else:
    indices = df[
      df[args.target_col].isna() | (df[args.target_col].astype(str).str.strip() == "")
    ].index.tolist()

  if not indices:
    print("No rows to translate.")
    sys.exit(0)

  print(f"Found {len(indices)} rows to translate.")

  # Prepare sentences
  sentences = []
  for idx in indices:
    text = str(df.at[idx, args.source_col])
    if args.strip_tags:
      text = strip_tags(text)
    sentences.append(text)

  # Load model and translate
  print("Loading translation model...")
  load_model()

  print(f"Translating {len(sentences)} sentences...")
  translations = []
  for i in tqdm(
    range(0, len(sentences), args.batch_size),
    desc="Translating",
    unit="batch",
  ):
    batch = sentences[i : i + args.batch_size]
    translations.extend(translate_lv_to_en_batch(batch))

  # Write translations back
  for idx, translation in zip(indices, translations):
    df.at[idx, args.target_col] = translation

  # Save output
  print(f"Saving to {output_path}...")
  if output_path.endswith(".csv"):
    df.to_csv(output_path, index=False)
  else:
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")

    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})

    # Preserve reasonable column widths
    for i, col in enumerate(df.columns):
      worksheet.set_column(i, i, 25, wrap_format)

    writer.close()

  print(f"Done. Translated {len(translations)} rows.")

if __name__ == "__main__":
  main()
