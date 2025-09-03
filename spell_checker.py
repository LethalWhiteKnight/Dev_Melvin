#!/usr/bin/env python3
from __future__ import annotations
import re
from collections import Counter
from typing import List, Tuple, Iterable, Dict, Set
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import words

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

# Initialize stemmer
stemmer = PorterStemmer()

# -----------------------------
# 1) Levenshtein DP (from scratch)
# -----------------------------

def levenshtein_matrix(a: str, b: str) -> List[List[int]]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            bj = b[j - 1]
            cost_sub = 0 if ai == bj else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost_sub  # substitute
            )
    return dp

def levenshtein_distance(a: str, b: str) -> int:
    return levenshtein_matrix(a, b)[-1][-1]

def backtrace_edits(a: str, b: str, dp: List[List[int]]) -> List[Tuple[str, str, str]]:
    edits: List[Tuple[str, str, str]] = []
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if a[i-1] == b[j-1] else 1):
            if a[i-1] == b[j-1]:
                edits.append(("MATCH", a[i-1], b[j-1]))
            else:
                edits.append(("SUB", a[i-1], b[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            edits.append(("DEL", a[i-1], ""))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            edits.append(("INS", "", b[j-1]))
            j -= 1
        else:
            raise RuntimeError("Invalid backtrace state")
    edits.reverse()
    return edits

def print_dp_matrix(a: str, b: str, dp: List[List[int]]) -> None:
    a_chars = [" "] + list(a)
    b_chars = [" "] + list(b)
    header = [" "] + b_chars
    col_width = max(3, max(len(str(x)) for row in dp for x in row) + 1)

    def cell(x):
        return str(x).rjust(col_width)

    print("".join(cell(h) for h in header))
    for i, row in enumerate(dp):
        line = [a_chars[i]] + row
        print("".join(cell(x) for x in line))


# ---------------------------------
# 2) Enhanced tokenizer & corpus loader with stemming
# ---------------------------------
TOKEN_RE = re.compile(r"[a-z]+")

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())

def get_stemmed_word(word: str) -> str:
    """Get the stemmed version of a word"""
    return stemmer.stem(word.lower())

def get_word_variations(word: str) -> Set[str]:
    """Get possible variations of a word (original, stemmed, and common suffixes)"""
    variations = {word.lower()}
    
    # Add stemmed version
    stemmed = get_stemmed_word(word)
    variations.add(stemmed)
    
    # Add common variations
    if word.endswith('s'):
        variations.add(word[:-1])  # Remove 's'
    if word.endswith('ing'):
        variations.add(word[:-3])  # Remove 'ing'
    if word.endswith('ed'):
        variations.add(word[:-2])  # Remove 'ed'
    if word.endswith('er'):
        variations.add(word[:-2])  # Remove 'er'
    if word.endswith('est'):
        variations.add(word[:-3])  # Remove 'est'
    if word.endswith('ly'):
        variations.add(word[:-2])  # Remove 'ly'
    
    return variations


# ---------------------------------
# 3) Enhanced Spell Checker with NLTK dictionary
# ---------------------------------
class SpellChecker:
    def __init__(self, corpus_text: str | None = None):
        # Initialize with NLTK words dictionary
        self.nltk_words = set(words.words())
        
        # Add custom corpus words
        if corpus_text is None:
            corpus_text = DEMO_CORPUS
        self.word_counts: Counter[str] = Counter(tokenize(corpus_text))
        
        # Combine NLTK words with corpus words
        self.vocab: Set[str] = self.nltk_words.union(set(self.word_counts))
        
        # Create stemmed vocabulary for better matching
        self.stemmed_vocab: Dict[str, Set[str]] = {}
        for word in self.vocab:
            stemmed = get_stemmed_word(word)
            if stemmed not in self.stemmed_vocab:
                self.stemmed_vocab[stemmed] = set()
            self.stemmed_vocab[stemmed].add(word)
        
        self.total: int = sum(self.word_counts.values()) or 1
        self.letters = "abcdefghijklmnopqrstuvwxyz"

    def known(self, words: Iterable[str]) -> Set[str]:
        return {w for w in words if w in self.vocab}

    def known_with_stemming(self, words: Iterable[str]) -> Set[str]:
        """Find known words including stemmed variations"""
        known_words = set()
        for word in words:
            # Check original word
            if word in self.vocab:
                known_words.add(word)
            
            # Check stemmed variations
            variations = get_word_variations(word)
            for variation in variations:
                if variation in self.vocab:
                    known_words.add(variation)
                
                # Check if stemmed version matches any words
                stemmed = get_stemmed_word(variation)
                if stemmed in self.stemmed_vocab:
                    known_words.update(self.stemmed_vocab[stemmed])
        
        return known_words

    def edits1(self, word: str) -> Set[str]:
        letters = self.letters
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        replaces = [L + c + (R[1:] if R else "") for L, R in splits if R for c in letters]
        inserts  = [L + c + R for L, R in splits for c in letters]
        return set(deletes + replaces + inserts)

    def edits2(self, word: str) -> Set[str]:
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}

    def candidates(self, word: str) -> Set[str]:
        # First check if the word is already correct
        if word in self.vocab:
            return {word}
        
        # Generate candidates with edit distance 1 and 2
        candidates = set()
        
        # Add original word if it exists in vocab
        candidates.update(self.known([word]))
        
        # Add edit distance 1 candidates
        candidates.update(self.known(self.edits1(word)))
        
        # Add edit distance 2 candidates
        candidates.update(self.known(self.edits2(word)))
        
        # Add stemmed variations
        candidates.update(self.known_with_stemming([word]))
        candidates.update(self.known_with_stemming(self.edits1(word)))
        candidates.update(self.known_with_stemming(self.edits2(word)))
        
        return candidates

    def distance(self, a: str, b: str) -> int:
        return levenshtein_distance(a, b)

    def score(self, misspelled: str, cand: str) -> Tuple[int, int, str]:
        d = self.distance(misspelled, cand)
        return (d, -self.word_counts[cand], cand)

    def suggest(self, word: str, k: int = 5) -> List[Tuple[str, int, int]]:
        cands = self.candidates(word)
        ranked = sorted(cands, key=lambda c: self.score(word, c))
        return [(c, self.distance(word, c), self.word_counts[c]) for c in ranked[:k]]

    def correct(self, word: str) -> str | None:
        cands = self.candidates(word)
        if not cands:
            return None
        return min(cands, key=lambda c: self.score(word, c))


# ---------------------------------
# 4) Demo / quick tests
# ---------------------------------
DEMO_CORPUS = (
    """
    Alice was beginning to get very tired of sitting by her sister on the bank,
    and of having nothing to do: once or twice she had peeped into the book her
    sister was reading, but it had no pictures or conversations in it.
    """
)

def _quick_unit_tests() -> None:
    pairs = [
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("intention", "execution", 5),
        ("", "", 0),
        ("abc", "abc", 0),
        ("a", "", 1),
        ("", "a", 1),
    ]
    for a, b, want in pairs:
        got = levenshtein_distance(a, b)
        assert got == want, f"levenshtein({a},{b})={got}, want {want}"
    print("✓ Basic Levenshtein unit tests passed.")

def _demo_backtrace(a: str, b: str) -> None:
    print(f"\nBacktrace demo: '{a}' → '{b}'")
    dp = levenshtein_matrix(a, b)
    print_dp_matrix(a, b, dp)
    edits = backtrace_edits(a, b, dp)
    print("Edits:")
    for op, x, y in edits:
        if op == "MATCH":
            print(f"  MATCH       {x} → {y}")
        elif op == "SUB":
            print(f"  SUBSTITUTE  {x} → {y}")
        elif op == "DEL":
            print(f"  DELETE      {x}")
        elif op == "INS":
            print(f"  INSERT         {y}")

def _demo_spellchecker() -> None:
    print("\nTraining enhanced spell checker with NLTK dictionary…")
    sc = SpellChecker(DEMO_CORPUS)
    tests = ["alcie", "conversatoin", "pictres", "rabit", "suddenly", "sleppey", "conversations", "sleeping", "beautifully"]
    for w in tests:
        suggestion = sc.correct(w)
        suggestions = sc.suggest(w, k=5)
        print(f"\nWord: {w}")
        print(f"  Best: {suggestion}")
        print("  Top-5:")
        for cand, dist, freq in suggestions:
            print(f"    {cand:>12}  dist={dist}  freq={freq}")

if __name__ == "__main__":
    _quick_unit_tests()
    _demo_backtrace("intention", "execution")
    _demo_spellchecker() 