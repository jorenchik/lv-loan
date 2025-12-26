
import argparse
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class Sample:
  words:  list[str]
  lemmas: list[str] 

class Corpus:
  samples: list[Sample] = []
  sample: Sample | None

  def __init__(self, input_vert) -> None:
  
    num_lines = sum(1 for _ in open(input_vert))
    inside_s = False

    with open(input_vert) as f:

      for line in tqdm(f, total=num_lines):

        line = line.strip()

        if line == "":
          continue

        if line == "<s>":
          inside_s = True
          self.sample = Sample([], []) 
          continue

        elif line == "</s>":
          inside_s = False 
          if self.sample:
            self.samples.append(self.sample)
          continue

        if inside_s:
          parts = line.split("\t")
          word, lemma = "", "" 

          if len(parts) >= 1:
            word = parts[0]

          if len(parts) >= 3:
            lemma = parts[2]

          if self.sample:
            self.sample.words.append(word)
            self.sample.lemmas.append(lemma)

  def search(self, str_: str, n: int = 1) \
    -> tuple[list[Sample], list[int]]:

    num_found: int = 0
    found: list[Sample] = []
    indices: list[int] = []

    for s in self.samples:
      
      if num_found >= n:
        break

      for i, lemma in enumerate(s.lemmas):
        if lemma.strip() == str_.strip():
          found.append(s)
          indices.append(i)
          num_found += 1
    
    return found, indices


def interactive(corpus: Corpus):
  
  while True:

    str_ = input("> ")
    found: Sample | None = None
    idx:   int    | None = None

    res = corpus.search(str_)
    found = res[0][0]
    idx   = res[1][0]

    if found:
      for i, w in enumerate(found.words):
        if idx == i:
          print("<L>", end="")
        print(w, end="")
        if idx == i:
          print("</L>", end=" ")
        else:
          print("", end=" ")
      print()

def main():

  parser = argparse.ArgumentParser(
    description="Simple .vert corpus parser"
  )
  parser.add_argument(
    "input_vert",
    type=str,
    help="Input corpora (.vert format)"
  )
  args = parser.parse_args()

  corpus = Corpus(args.input_vert)
  interactive(corpus)

if __name__ == "__main__":
  main()
