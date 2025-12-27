import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass(frozen=True)
class Lemma:
  written_form: str
  part_of_speech: str

@dataclass
class LexicalEntry:
  id: str
  lemma: Lemma
  senses: Dict[str, str] = field(default_factory=dict)

@dataclass
class Synset:
  id: str
  definition: Optional[str] = None
  members: List[str] = field(default_factory=list)

@dataclass
class SenseMatch:
  synset_id:  str
  definition: Optional[str]
  synonyms:   List[str]

@dataclass
class EntryMatch:
  pos:    str
  senses: List[SenseMatch]

@dataclass
class SearchResult:
  word:    str
  found:   bool
  entries: List[EntryMatch]

class WordNet:
  def __init__(self, file_path: str):
    self.entries: Dict[str, LexicalEntry] = {}
    self.synsets: Dict[str, Synset] = {}
    self._word_to_entries: Dict[str, List[str]] = {}
    # Trigger the parsing immediately during initialization
    self._load_from_lmf(file_path)

  def _load_from_lmf(self, file_path: str):
    """Internal memory-efficient streaming XML parser."""
    context = ET.iterparse(file_path, events=('end',))
    for _, elem in context:
      if elem.tag == 'LexicalEntry':
        entry_id = elem.get('id')
        lemma_elem = elem.find('Lemma')
        if entry_id and lemma_elem is not None:
          lemma = Lemma(
            written_form=lemma_elem.get('writtenForm', ''),
            part_of_speech=lemma_elem.get('partOfSpeech', '')
          )
          entry = LexicalEntry(id=entry_id, lemma=lemma)
          for sense in elem.findall('Sense'):
            s_id = sense.get('id')
            syn_id = sense.get('synset')
            if s_id and syn_id:
              entry.senses[s_id] = syn_id
          self.entries[entry_id] = entry
          word_key = lemma.written_form.lower()
          self._word_to_entries.setdefault(word_key, []).append(entry_id)
        elem.clear()

      elif elem.tag == 'Synset':
        syn_id = elem.get('id')
        if syn_id:
          members_attr = elem.get('members', '')
          members_list = members_attr.split() if members_attr else []
          defn_elem = elem.find('Definition')
          defn_text = defn_elem.text if defn_elem is not None else None
          synset = Synset(id=syn_id, definition=defn_text, members=members_list)
          self.synsets[syn_id] = synset
        elem.clear()

  def get_synonym_groups(self, word: str) -> SearchResult:
    """Retrieves structured SearchResult for a word."""
    query = word.strip().lower()
    e_ids = self._word_to_entries.get(query, [])
    
    if not e_ids:
      return SearchResult(word=word, found=False, entries=[])

    entry_matches = []
    for eid in e_ids:
      entry = self.entries[eid]
      senses_matches = []
      
      for syn_id in entry.senses.values():
        syn = self.synsets.get(syn_id)
        if not syn: 
          continue
        
        synonym_names = []
        for member_id in syn.members:
          member_entry = self.entries.get(member_id)
          if member_entry:
            name = member_entry.lemma.written_form
            if name.lower() != query:
              synonym_names.append(name)
        
        senses_matches.append(SenseMatch(
          synset_id=syn_id,
          definition=syn.definition,
          synonyms=sorted(list(set(synonym_names)))
        ))
        
      entry_matches.append(EntryMatch(
        pos=entry.lemma.part_of_speech,
        senses=senses_matches
      ))

    return SearchResult(word=word, found=True, entries=entry_matches)

def presentation_mode(wn: WordNet):
  print("\n" + "="*50)
  print("Latvian WordNet Synonym Search")
  print("Type 'exit' to quit.")
  print("="*50)

  while True:
    try:
      query = input("\nWord to search > ").strip()
      if query.lower() in ('exit', 'quit'):
        break
      if not query: 
        continue

      result = wn.get_synonym_groups(query)

      if not result.found:
        print(f"Error: The word '{query}' was not found in the dictionary.")
        continue

      print(f"\nWord '{result.word}' found ({len(result.entries)} variant[s]).")
      
      for entry in result.entries:
        print(f"\n--- Part of Speech: {entry.pos.upper()} ---")
        if not entry.senses:
          print("  No semantic senses defined for this variant.")
          continue

        for sense in entry.senses:
          print(f"\nMeaning: {sense.definition or 'No definition available'}")
          if sense.synonyms:
            print(f"Synonyms: {', '.join(sense.synonyms)}")
          else:
            print("Synonyms: No other synonyms linked to this specific meaning.")
      
      print("\n" + "-"*30)

    except KeyboardInterrupt:
      break

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Tezaurs WordNet LMF Synonym Parser")
  parser.add_argument("file", help="Path to the tezaurs_..._lmf.xml file")
  args = parser.parse_args()

  try:
    print(f"Loading {args.file}...")
    wordnet = WordNet(args.file)
    print(f"Loaded {len(wordnet.entries)} entries and {len(wordnet.synsets)} synsets.")
    presentation_mode(wordnet)
  except FileNotFoundError:
    print(f"Error: File '{args.file}' not found.")
  except Exception as e:
    print(f"Error initializing WordNet: {e}")
