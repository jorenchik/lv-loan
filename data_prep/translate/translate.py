import argparse
import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODELS = {
  "opus": "Helsinki-NLP/opus-mt-lv-en",
  "nllb": "facebook/nllb-200-distilled-600M",
}

DEFAULT_MODEL = os.environ.get("TRANSLATE_MODEL", "opus")

_tokenizer = None
_model = None
_model_type = None
_device = None


def load_model(model_key=None):
  global _tokenizer, _model, _model_type, _device

  if model_key is None and _model is not None:
    return _tokenizer, _model

  if model_key is None:
    model_key = DEFAULT_MODEL

  if model_key in MODELS:
    model_name = MODELS[model_key]
    model_type = model_key
  else:
    model_name = model_key
    model_type = "nllb" if "nllb" in model_key.lower() else "opus"

  if _model is not None and _model_type == model_type:
    return _tokenizer, _model

  _device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Loading model: {model_name} ({_device})")

  _tokenizer = AutoTokenizer.from_pretrained(model_name)
  _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

  if _device == "cuda":
    _model = _model.half()

  _model = _model.to(_device)
  _model.eval()
  _model_type = model_type

  return _tokenizer, _model


def translate_lv_to_en_batch(texts, model_key=None):
  """Translate list of Latvian texts to English."""
  if not texts:
    return []

  tok, mod = load_model(model_key)

  if _model_type == "nllb":
    tok.src_lang = "lvs_Latn"

  inputs = tok(texts, return_tensors="pt", padding=True, truncation=True).to(_device)

  generate_kwargs = {"max_length": 200}
  if _model_type == "nllb":
    generate_kwargs["forced_bos_token_id"] = tok.convert_tokens_to_ids("eng_Latn")

  with torch.no_grad():
    translated_tokens = mod.generate(**inputs, **generate_kwargs)

  return tok.batch_decode(translated_tokens, skip_special_tokens=True)


def translate_lv_to_en(text, model_key=None):
  """Translate single Latvian text to English."""
  if not text:
    return ""
  results = translate_lv_to_en_batch([text], model_key)
  return results[0] if results else ""


def main():
  parser = argparse.ArgumentParser(
    description="Latvian to English Translator (Opus-MT / NLLB)"
  )
  parser.add_argument(
    "--model", "-m",
    choices=list(MODELS.keys()),
    default=DEFAULT_MODEL,
    help=f"Model to use (default: {DEFAULT_MODEL})",
  )
  parser.add_argument(
    "--input", "-i",
    help="Path to text file with one sentence per line",
  )
  parser.add_argument(
    "sentences",
    nargs="*",
    help="Direct sentences to translate",
  )

  args = parser.parse_args()

  lines = []
  if args.input:
    with open(args.input, "r", encoding="utf-8") as f:
      lines = [line.strip() for line in f if line.strip()]
  elif args.sentences:
    lines = args.sentences
  else:
    if sys.stdin.isatty():
      print("Enter Latvian text (one per line, Ctrl+D to finish):")
    lines = [line.strip() for line in sys.stdin if line.strip()]

  if not lines:
    print("Error: No input text provided.", file=sys.stderr)
    sys.exit(1)

  results = translate_lv_to_en_batch(lines, args.model)

  for original, translated in zip(lines, results):
    print(f"LV: {original}")
    print(f"EN: {translated}")
    print("-" * 40)


if __name__ == "__main__":
  main()
