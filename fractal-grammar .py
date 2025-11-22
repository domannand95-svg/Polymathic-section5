"""
fractal_grammar_advanced.py

Enhanced Fractal Grammar with Yin-Yang Polarity Constraints and Phrase-Structure Recursion

Features:
- Complex nested fractal productions respecting symbolic grammar
- Polarity balancing with adjustable Yin-Yang harmony tolerance
- Phrase-structure-level transformations layered on base fractal expansions
- Tagged archetypal metadata output for transformer datasets
"""

import random
from archetypes import get_archetype_info, yin_yang_polarity

class FractalGrammarAdvanced:
    def __init__(self, rules, lexicon, polarity_tolerance=0.2):
        """
        rules: dict of nonterminals to list of expansions (each expansion is a list of symbols)
        lexicon: set/list of terminal symbols
        polarity_tolerance: maximum allowable imbalance in Yin-Yang polarity ratio in expansions
        """
        self.rules = rules
        self.lexicon = set(lexicon)
        self.polarity_tolerance = polarity_tolerance

    def calculate_sequence_polarity_score(self, sequence):
        """
        Calculate balance score between Yin and Yang polarities for given symbol sequence.
        Returns a float from 0 (no balance) to 1 (perfect balance)
        """
        yin_count = 0
        yang_count = 0
        for sym in sequence:
            meta = get_archetype_info(sym) or {}
            pol = meta.get('yin_yang')
            if pol == 'Yin':
                yin_count += 1
            elif pol == 'Yang':
                yang_count += 1
        total = yin_count + yang_count
        if total == 0:
            return 1.0  # No polarity symbols, consider balanced
        ratio = yin_count / total
        # Closer to 0.5 is more balanced, scale to [0,1]
        score = 1.0 - abs(0.5 - ratio) * 2
        return max(0, min(score, 1))

    def filter_expansions_by_polarity(self, expansions):
        """
        Filter candidate expansions based on polarity tolerance to filter imbalanced Yin/Yang expansions.
        """
        filtered = []
        for expansion in expansions:
            score = self.calculate_sequence_polarity_score(expansion)
            if score >= (1 - self.polarity_tolerance):
                filtered.append(expansion)
        return filtered or expansions  # Ensure never empty

    def phrase_transform(self, sequence):
        """
        Apply phrase-structure level transformations to symbolic sequences.
        This placeholder example swaps first two symbols if both terminals.
        """
        if len(sequence) > 1 and all(sym in self.lexicon for sym in sequence[:2]):
            return [sequence[1], sequence[0]] + sequence[2:]
        return sequence

    def expand_symbol(self, symbol, max_depth=6, current_depth=0, randomize=True):
        """
        Recursively expand a symbol into a sequence of terminals tagged with archetypes.
        Applies polarity constraints and phrase transformations.
        """
        if current_depth > max_depth:
            # Stop recursion
            return [(symbol, get_archetype_info(symbol) or {})]

        if symbol in self.lexicon:
            return [(symbol, get_archetype_info(symbol) or {})]

        if symbol not in self.rules or not self.rules[symbol]:
            return [(symbol, get_archetype_info(symbol) or {})]

        productions = self.rules[symbol]

        # Generate candidate expansions sequences
        candidate_expansions = []
        for prod in productions:
            expanded_seq = []
            for sym in prod:
                expanded_seq.extend(self.expand_symbol(sym, max_depth, current_depth+1, randomize))
            candidate_expansions.append([sym for sym, _ in expanded_seq])

        # Filter by polarity balance
        candidate_expansions = self.filter_expansions_by_polarity(candidate_expansions)

        # Select one expansion sequence
        chosen_seq = random.choice(candidate_expansions) if randomize else candidate_expansions[0]

        # Apply phrase transformations (optional)
        transformed_seq = self.phrase_transform(chosen_seq)

        # Return enriched sequence with metadata
        return [(sym, get_archetype_info(sym) or {}) for sym in transformed_seq]

    def generate(self, start_symbol='S', max_depth=6, randomize=True):
        """
        Generate a fully expanded, polarity balanced fractal grammar sequence with archetypal tagging.
        """
        return self.expand_symbol(start_symbol, max_depth, 0, randomize)

# -----------------------------------
# Complex fractal grammar rules that include recursion and polarity balancing
# -----------------------------------

advanced_rules = {
    'S': [['NP', 'VP'], ['VP', 'NP']],  # sentence can invert phrase order fractally
    'NP': [['Det', 'N'], ['Adj', 'N']],  # noun phrase
    'VP': [['V', 'NP'], ['V']],           # verb phrase
    'Det': [['ה'], ['א']],                 # Hebrew definite articles/examples in lexicon
    'N': [['א'], ['ב'], ['ג']],            # Nouns (terminals)
    'Adj': [['ד'], ['ה']],                 # Adjectives (terminals)
    'V': [['ו'], ['ז']],                   # Verbs (terminals)
}

advanced_lexicon = ['ה', 'א', 'ב', 'ג', 'ד', 'ו', 'ז']

# -----------------------------------
# Example usage
# -----------------------------------

if __name__ == "__main__":
    fg_adv = FractalGrammarAdvanced(advanced_rules, advanced_lexicon, polarity_tolerance=0.3)
    seq = fg_adv.generate('S', max_depth=5, randomize=True)
    print("Advanced Fractal Grammar Generated Sequence with Archetypal Metadata:")
    for sym, meta in seq:
        gender = meta.get('gender', 'unknown')
        polarity = meta.get('polarity', 'neutral')
        element = meta.get('element', 'none')
        yin_yang = meta.get('yin_yang', 'none')
        print(f"Symbol: {sym:2} | Gender: {gender:6} | Polarity: {polarity:7} | Element: {element:5} | YinYang: {yin_yang}")
        """
sigil_maker.py

Sigil Making Module

- Input: Intent phrase (string)
- Maps letters to archetypal glyph tokens (using your archetypes/alphabet mapping)
- Condenses repeated symbols and applies polarity balancing
- Generates fractal-inspired parametric glyph data structures
- Provides scalable visual and parametric outputs for synthesis
"""

import string
import matplotlib.pyplot as plt
import numpy as np

# Simple archetypal letter-to-symbol map (extend from your archetypes.py)
letter_to_glyph = {
    'a': 'א', 'b': 'ב', 'c': 'ג', 'd': 'ד', 'e': 'ה',
    'f': 'ו', 'g': 'ז', 'h': 'ח', 'i': 'ט', 'j': 'י',
    'k': 'כ', 'l': 'ל', 'm': 'מ', 'n': 'נ', 'o': 'ס',
    'p': 'פ', 'q': 'צ', 'r': 'ק', 's': 'ר', 't': 'ש',
    'u': 'ת', 'v': 'ט', 'w': 'וו', 'x': 'קס', 'y': 'יי', 'z': 'ז'
}

def normalize_phrase(phrase):
    """
    Normalize input string: lowercase, remove spaces/punctuation
    """
    allowed = string.ascii_lowercase
    return ''.join(ch for ch in phrase.lower() if ch in allowed)

def phrase_to_glyph_sequence(phrase):
    """
    Map input phrase letters to archetypal glyph tokens.
    """
    normalized = normalize_phrase(phrase)
    glyphs = [letter_to_glyph.get(ch, '') for ch in normalized]
    # Flatten composite glyphs like 'וו'
    flat_glyphs = []
    for g in glyphs:
        if len(g) > 1:
            flat_glyphs.extend(list(g))
        else:
            flat_glyphs.append(g)
    return flat_glyphs

def condense_glyphs(glyph_sequence):
    """
    Remove duplicate consecutive glyphs and apply simple polarity balance adjustments.
    """
    if not glyph_sequence:
        return []

    condensed = [glyph_sequence[0]]

    for g in glyph_sequence[1:]:
        if g != condensed[-1]:
            condensed.append(g)
    return condensed

# ---------------------------------
# Fractal-like parametric glyph primitive generator (simplified)
# ---------------------------------

def glyph_parametrization(glyph, index, total, radius=1.0):
    """
    Parametrize glyph placement on a circle with fractal-inspired offset.
    """
    angle = 2 * np.pi * index / total
    x = radius * np.cos(angle) * (1 + 0.2*np.sin(index*3))
    y = radius * np.sin(angle) * (1 + 0.2*np.cos(index*5))
    size = 0.1 + 0.05 * (index % 3)
    return {'glyph': glyph, 'x': x, 'y': y, 'size': size}

def generate_sigil_params(glyph_sequence):
    """
    Generate parametric data for fractal visualization of the sigil.
    Returns list of dicts with position and size for each glyph token.
    """
    total = len(glyph_sequence)
    params = []
    for i, glyph in enumerate(glyph_sequence):
        params.append(glyph_parametrization(glyph, i, total))
    return params

# ---------------------------------
# Visualization Example
# ---------------------------------

def visualize_sigil(params):
    """
    Render parametric sigil glyphs as simple circles with glyph labels using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')

    for p in params:
        circle = plt.Circle((p['x'], p['y']), p['size'], color='purple', alpha=0.6)
        ax.add_artist(circle)
        ax.text(p['x'], p['y'], p['glyph'], fontsize=12, ha='center', va='center', color='white')

    plt.show()

# ---------------------------------
# Example Usage
# ---------------------------------

if __name__ == "__main__":
    input_phrase = "Sacred Protection"
    glyph_seq = phrase_to_glyph_sequence(input_phrase)
    condensed_seq = condense_glyphs(glyph_seq)
    sigil_params = generate_sigil_params(condensed_seq)

    print("Input phrase:", input_phrase)
    print("Glyph sequence:", condensed_seq)

    visualize_sigil(sigil_params)
    # --------------------------------------------
# Section Z: Fractal Grammar Integration
# --------------------------------------------

# Reference to your fractal grammar rules dictionary (assuming it's in fractal_grammar_advanced.py or similar)
# Here we include symbolic keys to locate it by name or import externally as needed.

fractal_grammar_rules_reference = "fractal_grammar_advanced.fractal_grammar_rules"

# Example polarity tolerance parameter linked to fractal grammar recursive expansions
fractal_polarity_tolerance = 0.3

def get_fractal_grammar_rules():
    """
    Access fractal grammar rules dictionary from external module or internal reference.
    If fractal grammar rules are defined here, return directly; otherwise import dynamically.
    """
    try:
        from fractal_grammar_advanced import fractal_grammar_rules
        return fractal_grammar_rules
    except ImportError:
        # Or return empty/default if not available
        return {}

def get_fractal_polarity_tolerance():
    """
    Returns the polarity tolerance threshold used to balance fractal grammar expansions.
    """
    return fractal_polarity_tolerance

# Optional helper to validate fractal grammar rules coherence with archetypes
def validate_fractal_grammar_terms(archetype_symbols):
    """
    Validates that fractal grammar terminal symbols map to known archetypal symbols.
    archetype_symbols: set/list of all known archetype symbols in this archetypes.py
    Returns list of unknown symbols.
    """
    rules = get_fractal_grammar_rules()
    terminals = rules.get("terminals", {})
    unknowns = [term for term in terminals.keys() if terminals[term]['symbol'] not in archetype_symbols]
    return unknowns

# Example usage - this code can be commented out or run during tests:
if __name__ == "__main__":
    archetype_syms = {v['symbol'] for v in globals().get("planets", {}).values()}
    unknown_terms = validate_fractal_grammar_terms(archetype_syms)
    if unknown_terms:
        print(f"Warning: Unknown terminal symbols in fractal grammar: {unknown_terms}")
    else:
        print("Fractal grammar terminals validated against archetypes.")
        # fractal_grammar_advanced.py (enhanced sections)

# Import archetypes and API functions
from archetypes import (
    get_archetype,
    combined_filter,
    get_fractal_polarity_tolerance,
    fractal_circles  # Or appropriate dict name if different
)

import random

# Utility for polarity mapping
POLARITY_MAP = {"Yang": 1, "Yin": -1, "Balanced": 0}

def get_symbol_metadata(symbol):
    """
    Retrieves archetypal metadata for a given grammar symbol.
    Searches fractal grammar terminals or archetypes broadly.
    """
    # Check fractal grammar terminals first (assumed fractal_grammar_rules dict)
    if symbol in fractal_grammar_rules.get("terminals", {}):
        return fractal_grammar_rules["terminals"][symbol]
    # Otherwise, attempt a broader archetype lookup by symbol across categories:
    for category in ['planets', 'metals', 'magick_circles', 'fractal_patterns']:
        filtered = combined_filter()
        for _, name, attrs in filtered:
            if attrs.get("symbol") == symbol:
                return attrs
    # Default fallback
    return {"symbol": symbol, "element": None, "polarity": "Balanced"}

def polarity_sum(sequence):
    """
    Calculate cumulative polarity sum of a sequence of archetypal tokens.
    """
    total = 0
    for token in sequence:
        pol_val = POLARITY_MAP.get(token.get('polarity', 'Balanced'), 0)
        total += pol_val
    return total

def is_polarity_balanced(sequence, tolerance=None):
    """
    Checks if polarity sum is within the allowed tolerance threshold.
    """
    if tolerance is None:
        tolerance = get_fractal_polarity_tolerance()
    return abs(polarity_sum(sequence)) <= tolerance

def expand_symbol(symbol, depth=0, max_depth=5):
    """
    Enhanced recursive fractal grammar expansion:
    - Queries archetypal metadata
    - Balances polarity dynamically per expansion
    - Optionally branches on archetype subclass (e.g., magick circle memberships)
    - Enriches terminals with multimodal metadata for downstream use
    """
    if depth > max_depth:
        terminal_meta = get_symbol_metadata(symbol)
        return [terminal_meta]

    if symbol in fractal_grammar_rules.get("terminals", {}):
        return [fractal_grammar_rules["terminals"][symbol]]

    productions = fractal_grammar_rules.get(symbol, [])
    # Dynamic filtering of productions for polarity balance:
    valid_productions = []
    for prod in productions:
        expanded_seq = []
        for sym in prod:
            expanded_seq.extend(expand_symbol(sym, depth + 1, max_depth))
        if is_polarity_balanced(expanded_seq):
            valid_productions.append(prod)

    # Fallback: if no balanced productions found, use all
    if not valid_productions:
        valid_productions = productions

    selected_prod = random.choice(valid_productions)

    # Recursive expansion of the selected production
    result = []
    for sym in selected_prod:
        result.extend(expand_symbol(sym, depth + 1, max_depth))

    # Branching example: inject magick circle segment metadata if applicable
    # This requires additional logic if 'symbol' matches a magick circle label
    if symbol in fractal_circles:
        circle_data = fractal_circles[symbol]
        # Example: enrich result with circle segment info or mark segment context
        for token in result:
            token['magick_circle'] = circle_data.get('name')
    
    # Return enriched hierarchical output sequence for multimodal use
    return result

# === Simple test cases for backward compatibility and new functionality ===
def test_expand_symbol():
    print("Testing fractal grammar expansion with archetypal integration...")
    seq = expand_symbol("S", max_depth=3)
    print("Expansion result:")
    for token in seq:
        print(f"Symbol: {token.get('symbol')}, Element: {token.get('element')}, Polarity: {token.get('polarity')}, MagickCircle: {token.get('magick_circle', 'None')}")

    assert is_polarity_balanced(seq), "Polarity balance test failed!"
    print("Polarity balance test passed.")

if __name__ == "__main__":
    test_expand_symbol()
    def expand_symbol(symbol, archetypes, max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        archetype = archetype_by_symbol(symbol, archetypes)
        if archetype:
            return [{
                "symbol": archetype["symbol"],
                "polarity": archetype.get("polarity", "Balanced"),
                "element": archetype.get("element", "None"),
                "gender": archetype.get("gender", "neutral")
            }]
        else:
            return [{
                "symbol": symbol,
                "polarity": "Balanced",
                "element": "None",
                "gender": "neutral"
            }]
    # Recursive fractal expansion: example rule for symbol "S"
    if symbol == "S":
        expanded = []
        for sub_symbol in ["א", "ב", "ג"]:
            expanded.extend(expand_symbol(sub_symbol, archetypes, max_depth, current_depth + 1))
        return expanded
    else:
        return [{
            "symbol": symbol,
            "polarity": "Balanced",
            "element": "None",
            "gender": "neutral"
        }]