
import subprocess
import math
import os
import tempfile
import sys
import argparse
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_CQP_BIN = "cqp" 
DEFAULT_CQP_DIR = "/home/jorenchik/cwb"
DEFAULT_CORPUS = "EMUARI"
DEFAULT_RESULTS = 200

@dataclass
class Token:
    word: str
    pos: str
    lemma: str

@dataclass
class CQPResult:
    cqp_id: int              
    tokens: List[Token]      
    match_index: int         
    
    @property
    def text(self):
        return " ".join([t.word for t in self.tokens])

    @property
    def target_word(self):
        if 0 <= self.match_index < len(self.tokens):
            return self.tokens[self.match_index]
        return None

def parse_cqp_line(line: str) -> Optional[CQPResult]:
    """
    Parses a single line of standard CQP ASCII output.
    """
    try:
        id_part, content_part = line.split(':', 1)
        cqp_id = int(id_part.strip())
    except ValueError:
        return None 

    raw_tokens = content_part.strip().split()
    parsed_tokens = []
    match_index = -1
    
    for i, rt in enumerate(raw_tokens):

        if rt.startswith('<'):
            rt = rt[1:]
            if match_index == -1: 
              match_index = i
        
        if rt.endswith('>'):
            rt = rt[:-1]
            
        parts = rt.rsplit('/', 2)
        
        if len(parts) == 3:
            w, p, lemma = parts
        elif len(parts) == 2:
            w, p = parts
            lemma = w 
        else:
            w = rt
            p = "UNK"
            lemma = w

        parsed_tokens.append(
          Token(word=w, pos=p, lemma=lemma)
        )

    return CQPResult(
        cqp_id=cqp_id,
        tokens=parsed_tokens,
        match_index=match_index
    )

def query_cqp(corpus, query, limit, cqp_bin, cqp_dir):
    registry = f"{cqp_dir}/registry"
    
    commands = [
        f"{corpus};",
        "set Context 1 s;",
        "set PrintMode ascii;",
        "show -pos -lemma;",
        "show +pos +lemma;",
        "set PrintOptions noheader;",
        f"Results = {query};",
        f"reduce Results to {limit};",
        "cat Results;"
    ]

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write("\n".join(commands))
        temp_file_path = tf.name

    try:
        process = subprocess.Popen(
            [cqp_bin, "-r", registry, "-S", "-f", temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cqp_dir, 
        )
        stdout, _ = process.communicate()
        return stdout
    except FileNotFoundError:
        print(f"Error: Could not find CQP binary at '{cqp_bin}'", file=sys.stderr)
        return ""
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def gaussian(x, mu, sigma):
    return math.exp(-pow(x - mu, 2) / (2 * pow(sigma, 2)))

def score_sentence(res: CQPResult) -> float:
    tokens = res.tokens
    if not tokens: 
      return 0.0
    
    s_len = len(tokens)
    
    # 1. Length 
    score_len = gaussian(s_len, 18, 8)
    
    # 2. Capitalization 
    first_token = tokens[0]
    score_caps = 0.1
    if len(first_token.word) >= 1:
        first_char = first_token.word[0]
        score_caps = 1.0 if first_char.isupper() or first_char in '"\'“' else 0.1

    # 3. Position
    if res.match_index == 0 or res.match_index == s_len - 1:
        score_pos = 0.5
    else:
        score_pos = 1.0

    # 4. Prefer non-capitalized words (non proper nouns)
    target = res.target_word
    score_target_case = 1.0
    if target and res.match_index > 0:
        # If inside sentence and starts with Uppercase 
        #    -> likely a Name -> Penalize
        if len(target.word) >= 1 and target.word[0].isupper():
            score_target_case = 0.2

    # 5. Non-word features 
    n_punct = 0
    n_digit = 0
    for t in tokens:
        if t.pos.startswith('z') or not t.word.isalnum():
            n_punct += 1
        elif t.word.isdigit():
            n_digit += 1

    punct_ratio = n_punct / s_len
    score_punct = 1.0 if punct_ratio < 0.2 else max(0, 1.0 - (punct_ratio * 2))

    digit_ratio = n_digit / s_len
    score_digit = 1.0 if digit_ratio < 0.1 else 0.2

    # 6. Garbage/Clenliness 
    is_clean = 1.0
    # if s_len > 2 and tokens[1].word in [':', '-'] and tokens[0].word.replace(':','').isdigit():
    #     is_clean = 0.0
    text_blob = "".join([t.word for t in tokens])
    if text_blob.count('(') != text_blob.count(')'):
        is_clean *= 0.8
    if text_blob.count('"') % 2 != 0:
        is_clean *= 0.8

    # Combined score.
    base_score = (
        (score_len * 0.25) + 
        (score_punct * 0.25) + 
        (score_caps * 0.2) + 
        (score_pos * 0.1) +
        (score_target_case * 0.2) 
    )
    final_score = base_score * score_digit * is_clean

    return final_score

def main():

    parser = argparse.ArgumentParser(description="Query CQP and score results.")
    parser.add_argument(
      "lemma",
      help="The target lemma to search for"
    )
    parser.add_argument(
      "--corpus",
      default=DEFAULT_CORPUS,
      help=f"Corpus name (default: {DEFAULT_CORPUS})"
    )
    parser.add_argument(
      "--limit",
      type=int,
      default=DEFAULT_RESULTS,
      help=f"Max results (default: {DEFAULT_RESULTS})"
    )
    parser.add_argument(
      "--cqp-bin",
      default=DEFAULT_CQP_BIN,
      help="Path to cqp binary"
    )
    parser.add_argument(
      "--cqp-dir",
      default=DEFAULT_CQP_DIR,
      help="Path to cwb directory"
    )
    args = parser.parse_args()

    search_query = f'[lemma="{args.lemma}"]'
    print(f"--- Querying {args.corpus} for: {search_query} ---")
    
    raw_output = query_cqp(
        args.corpus, 
        search_query, 
        args.limit, 
        args.cqp_bin, 
        args.cqp_dir
    )
    
    results = []
    if raw_output:
        for line in raw_output.split('\n'):
            if not line.strip(): 
              continue
            parsed = parse_cqp_line(line)
            if parsed:
                results.append(parsed)

    print(f"Parsed {len(results)} results.\n")

    scored_results = []
    for s in results:
        score = score_sentence(s)
        scored_results.append((score, s))
    
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"Top 5 Results for '{args.lemma}':")
    print("-" * 60)
    for score, res in scored_results[:5]:
        sent_str = ""
        for i, t in enumerate(res.tokens):
            if i == res.match_index:
                sent_str += f"\033[92m>>{t.word}<<\033[0m " 
            else:
                sent_str += f"{t.word} "
        
        print(f"Score: {score:.4f} | ID: {res.cqp_id}")
        print(sent_str.strip())
        print("-" * 60)

if __name__ == "__main__":
    main()
