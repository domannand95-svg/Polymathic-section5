"""
archetypes.py

Comprehensive Archetypal Symbology Module
Includes:
- Planetary archetypes with elements, metals, angels, genders
- Angelic correspondences
- Tarot Major Arcana and Jungian archetypes
- Polarity, Yin-Yang, gender, rotation, magnetic poles, hot-cold dualities
- Feng Shui elements and geomancy
- Universal harmony solfeggio frequencies
- Dynamic energy-gender meta-encoding and decoding utilities
"""

# --------------------------------------------
# Section 1: Planetary Archetypes with Elements & Metals and Gender
# --------------------------------------------

planetary_archetypes = {
    'Sun': {
        'gender': 'male',
        'polarity': +1,
        'yin_yang': 'Yang',
        'element': 'Fire',
        'metal': 'Gold',
        'angel': 'Michael',
        'frequency': 528,
        'notes': 'Vitality, creativity, leadership'
    },
    'Moon': {
        'gender': 'female',
        'polarity': -1,
        'yin_yang': 'Yin',
        'element': 'Water',
        'metal': 'Silver',
        'angel': 'Gabriel',
        'frequency': 396,
        'notes': 'Intuition, emotions, cycles'
    },
    'Mars': {
        'gender': 'male',
        'polarity': +1,
        'yin_yang': 'Yang',
        'element': 'Fire',
        'metal': 'Iron',
        'angel': 'Samael',
        'frequency': 417,
        'notes': 'Courage, aggression, energy'
    },
    'Venus': {
        'gender': 'female',
        'polarity': -1,
        'yin_yang': 'Yin',
        'element': 'Earth',
        'metal': 'Copper',
        'angel': 'Haniel',
        'frequency': 639,
        'notes': 'Love, beauty, harmony'
    },
    'Mercury': {
        'gender': 'male',
        'polarity': +1,
        'yin_yang': 'Yang',
        'element': 'Air',
        'metal': 'Mercury',
        'angel': 'Raphael',
        'frequency': 741,
        'notes': 'Communication, intellect, travel'
    },
    'Jupiter': {
        'gender': 'male',
        'polarity': +1,
        'yin_yang': 'Yang',
        'element': 'Air',
        'metal': 'Tin',
        'angel': 'Zadkiel',
        'frequency': 852,
        'notes': 'Expansion, wisdom, abundance'
    },
    'Saturn': {
        'gender': 'male',
        'polarity': -1,
        'yin_yang': 'Yin',
        'element': 'Earth',
        'metal': 'Lead',
        'angel': 'Cassiel',
        'frequency': 174,
        'notes': 'Discipline, limitations, time'
    },
}

# --------------------------------------------
# Section 2: Angelic Correspondences
# --------------------------------------------

angelic_archetypes = {
    'Michael': {
        'gender': 'male',
        'polarity': +1,
        'domains': ['Protection', 'Courage', 'Sun', 'Fire'],
    },
    'Gabriel': {
        'gender': 'female',
        'polarity': -1,
        'domains': ['Revelation', 'Communication', 'Moon', 'Water'],
    },
    'Raphael': {
        'gender': 'male',
        'polarity': +1,
        'domains': ['Healing', 'Travel', 'Mercury', 'Air'],
    },
    'Uriel': {
        'gender': 'male',
        'polarity': +1,
        'domains': ['Wisdom', 'Earth', 'Transformation'],
    },
    'Haniel': {
        'gender': 'female',
        'polarity': -1,
        'domains': ['Grace', 'Love', 'Venus', 'Earth'],
    },
    'Samael': {
        'gender': 'male',
        'polarity': +1,
        'domains': ['Judgment', 'Mars', 'Fire'],
    },
}

# --------------------------------------------
# Section 3: Tarot Major Arcana with Gender and Polarity
# --------------------------------------------

tarot_major_arcana = {
    'The Fool': {'number': 0, 'gender': 'male', 'polarity': +1, 'element': 'Air', 'notes': 'Beginnings, innocence'},
    'The Magician': {'number': 1, 'gender': 'male', 'polarity': +1, 'element': 'Air', 'notes': 'Willpower, manifestation'},
    'The High Priestess': {'number': 2, 'gender': 'female', 'polarity': -1, 'element': 'Water', 'notes': 'Intuition, mystery'},
    'The Empress': {'number': 3, 'gender': 'female', 'polarity': -1, 'element': 'Earth', 'notes': 'Fertility, abundance'},
    'The Emperor': {'number': 4, 'gender': 'male', 'polarity': +1, 'element': 'Fire', 'notes': 'Authority, stability'},
    'The Hierophant': {'number': 5, 'gender': 'male', 'polarity': +1, 'element': 'Earth', 'notes': 'Tradition, wisdom'},
    'The Lovers': {'number': 6, 'gender': 'dual', 'polarity': 0, 'element': 'Air', 'notes': 'Union, choice'},
    'The Chariot': {'number': 7, 'gender': 'male', 'polarity': +1, 'element': 'Water', 'notes': 'Victory, determination'},
    'Strength': {'number': 8, 'gender': 'female', 'polarity': -1, 'element': 'Fire', 'notes': 'Courage, compassion'},
    'The Hermit': {'number': 9, 'gender': 'male', 'polarity': +1, 'element': 'Earth', 'notes': 'Solitude, reflection'},
    # ... extend remaining cards as needed
}

# --------------------------------------------
# Section 4: Jungian Archetypes Mapping
# --------------------------------------------

jungian_archetypes = {
    'Self': {'description': 'The integrated whole; unity of unconscious and conscious', 'gender': 'dual', 'polarity': 0},
    'Shadow': {'description': 'Unrecognized or suppressed part of self', 'gender': 'varied', 'polarity': -1},
    'Anima': {'description': 'Feminine inner personality in the male psyche', 'gender': 'female', 'polarity': -1},
    'Animus': {'description': 'Masculine inner personality in the female psyche', 'gender': 'male', 'polarity': +1},
    'Persona': {'description': 'The social mask or role', 'gender': 'neutral', 'polarity': 0},
}

# --------------------------------------------
# Section 5: Polarity, Yin-Yang and Harmony Functions
# --------------------------------------------

def yin_yang_polarity(value):
    if value > 0:
        return 'Yang'
    elif value < 0:
        return 'Yin'
    else:
        return 'Balanced'

def calculate_balance_ratio(sequence):
    yin_count = sum(1 for item in sequence if item.get('yin_yang', '') == 'Yin')
    yang_count = sum(1 for item in sequence if item.get('yin_yang', '') == 'Yang')
    total = yin_count + yang_count
    if total == 0:
        return 1.0  # Balanced when no polarity given
    ratio = yin_count / total
    return ratio

def harmonic_coherence_score(sequence):
    ratio = calculate_balance_ratio(sequence)
    return 1.0 - abs(0.5 - ratio) * 2

# --------------------------------------------
# Section 6: Feng Shui Elements and Directions
# --------------------------------------------

feng_shui_elements = {
    'East': {'element': 'Wood', 'color': 'Green', 'season': 'Spring', 'direction_degrees': 90},
    'South': {'element': 'Fire', 'color': 'Red', 'season': 'Summer', 'direction_degrees': 180},
    'West': {'element': 'Metal', 'color': 'White', 'season': 'Autumn', 'direction_degrees': 270},
    'North': {'element': 'Water', 'color': 'Black', 'season': 'Winter', 'direction_degrees': 0},
    'Center': {'element': 'Earth', 'color': 'Yellow', 'season': 'Late Summer', 'direction_degrees': None},
}

# --------------------------------------------
# Section 7: Universal Harmony - Solfeggio Frequencies
# --------------------------------------------

universal_frequencies = {
    '396Hz': {'description': 'Liberating Guilt and Fear', 'associated_planet': 'Moon', 'archetype': 'Gabriel'},
    '417Hz': {'description': 'Undoing Situations and Facilitating Change', 'associated_planet': 'Mars', 'archetype': 'Samael'},
    '528Hz': {'description': 'Transformation and Miracles', 'associated_planet': 'Sun', 'archetype': 'Michael'},
    '639Hz': {'description': 'Connecting/Relationships', 'associated_planet': 'Venus', 'archetype': 'Haniel'},
    '741Hz': {'description': 'Awakening Intuition', 'associated_planet': 'Mercury', 'archetype': 'Raphael'},
    '852Hz': {'description': 'Returning to Spiritual Order', 'associated_planet': 'Jupiter', 'archetype': 'Zadkiel'}
}

# --------------------------------------------
# Section 8: Dynamic Energy-Gender Meta-Encoding Class
# --------------------------------------------

class EnergyGenderMeta:
    """
    Encodes multi-faceted energy and gender polarity metadata
    with support for continuous, dynamic, and interchangeable decoding.
    """

    def __init__(self, yin_yang=None, gender=None, thermal=None, rotation=None, magnetic_pole=None):
        self.yin_yang = yin_yang    # 'Yin' or 'Yang'
        self.gender = gender        # 'male' or 'female'
        self.thermal = thermal      # 'hot' or 'cold'
        self.rotation = rotation    # 'clockwise' or 'anticlockwise'
        self.magnetic_pole = magnetic_pole  # 'north' or 'south'

    def to_dict(self):
        return {
            'yin_yang': self.yin_yang,
            'gender': self.gender,
            'thermal': self.thermal,
            'rotation': self.rotation,
            'magnetic_pole': self.magnetic_pole,
        }

    def invert_polarities(self):
        invert_map = {
            'Yin': 'Yang', 'Yang': 'Yin',
            'male': 'female', 'female': 'male',
            'hot': 'cold', 'cold': 'hot',
            'clockwise': 'anticlockwise', 'anticlockwise': 'clockwise',
            'north': 'south', 'south': 'north',
        }
        self.yin_yang = invert_map.get(self.yin_yang, self.yin_yang)
        self.gender = invert_map.get(self.gender, self.gender)
        self.thermal = invert_map.get(self.thermal, self.thermal)
        self.rotation = invert_map.get(self.rotation, self.rotation)
        self.magnetic_pole = invert_map.get(self.magnetic_pole, self.magnetic_pole)

    def as_coherent_polarity(self):
        polarity_map = {
            'Yin': -1, 'Yang': +1,
            'male': +1, 'female': -1,
            'hot': +1, 'cold': -1,
            'clockwise': +1, 'anticlockwise': -1,
            'north': +1, 'south': -1,
        }
        attributes = [self.yin_yang, self.gender, self.thermal, self.rotation, self.magnetic_pole]
        polarity_values = [polarity_map.get(attr) for attr in attributes if attr in polarity_map]
        if not polarity_values:
            return 0
        return 1 if sum(polarity_values) >= 0 else -1

    def dynamic_decode(self, target_mode):
        mapping = {
            'feng_shui': lambda: 'Yang' if self.yin_yang == 'Yang' else 'Yin',
            'gender': lambda: self.gender,
            'thermal': lambda: self.thermal,
            'rotation': lambda: self.rotation,
            'magnetic': lambda: self.magnetic_pole,
        }
        if target_mode not in mapping:
            raise ValueError(f"Unsupported target decoding mode: {target_mode}")
        return mapping[target_mode]()

# --------------------------------------------
# Section 9: Utilities for Archetype Lookup and Gender Checks
# --------------------------------------------

def get_archetype_info(name):
    for collection in [planetary_archetypes, angelic_archetypes, tarot_major_arcana, jungian_archetypes]:
        if name in collection:
            return collection[name]
    return None

def is_male_archetype(name):
    info = get_archetype_info(name)
    if info:
        return info.get('gender', '').lower() == 'male'
    return False

def is_female_archetype(name):
    info = get_archetype_info(name)
    if info:
        return info.get('gender', '').lower() == 'female'
    return False

def archetypal_polarity(name):
    info = get_archetype_info(name)
    if info:
        return info.get('polarity', 0)
    return 0

# --------------------------------------------
# End of archetypes.py
"""
magick_circle_utils.py

Utilities for Fractal Glyph Generation and Ritual Scripting Using Magick Circle Data
"""

import math
import matplotlib.pyplot as plt
from archetypes import magick_circles  # Assuming appended to archetypes.py

# ----------------------------------------
# Fractal Glyph Generator for Magick Circles
# ----------------------------------------

def generate_magick_circle_glyph(circle_name):
    """
    Generates a fractal glyph visualization of a magick circle archetype.
    This simple example draws circle segments and places guardians/archetypes.
    """
    if circle_name not in magick_circles:
        raise ValueError(f"Magick circle '{circle_name}' not found.")

    circle = magick_circles[circle_name]
    segments = circle['segments']
    num_segments = len(segments)

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw={'aspect': 'equal'})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # Draw outer circle
    circle_outline = plt.Circle((0,0), 1, fill=False, linewidth=2, color='black')
    ax.add_artist(circle_outline)

    angle_per_segment = 2 * math.pi / num_segments
    for i, (segment_name, segment) in enumerate(segments.items()):
        start_angle = i * angle_per_segment
        end_angle = start_angle + angle_per_segment

        # Draw segment wedge
        wedge = plt.Polygon([
            (0,0),
            (math.cos(start_angle), math.sin(start_angle)),
            (math.cos(end_angle), math.sin(end_angle))
        ], color='lightgray', alpha=0.3)
        ax.add_artist(wedge)

        # Place guardian/archetype text near segment center
        text_angle = (start_angle + end_angle) / 2
        text_radius = 0.7
        x_text = text_radius * math.cos(text_angle)
        y_text = text_radius * math.sin(text_angle)

        guardian = segment.get('guardian') or segment.get('archetype') or segment.get('spirit_guide', 'Unknown')
        element = segment.get('element', '')
        label = f"{guardian}
{element}"

        ax.text(x_text, y_text, label, fontsize=10, ha='center', va='center')

    ax.set_title(circle['name'], fontsize=14, weight='bold')
    plt.show()


# ----------------------------------------
# Ritual Scripting Scaffold
# ----------------------------------------

class RitualSequence:
    def __init__(self, circle_name):
        if circle_name not in magick_circles:
            raise ValueError(f"Magick circle '{circle_name}' not found.")
        self.circle = magick_circles[circle_name]
        self.steps = []

    def add_step(self, description, invocation_targets=None, function=None):
        """
        Add a step to the ritual.
        - description: text instruction for the step
        - invocation_targets: list of keys in circle 'segments' to invoke
        - function: optional callable to execute for this step (e.g., sound or light control)
        """
        self.steps.append({
            'description': description,
            'targets': invocation_targets or [],
            'function': function
        })

    def run(self):
        """
        Execute the ritual stepwise, invoking functions and printing instructions.
        """
        for i, step in enumerate(self.steps):
            print(f"Step {i+1}: {step['description']}")
            if step['targets']:
                for target in step['targets']:
                    segment = self.circle['segments'].get(target, {})
                    guardian = segment.get('guardian') or segment.get('archetype') or segment.get('spirit_guide', 'Unknown')
                    print(f" - Invoking {guardian} at {target} segment.")
            if step['function']:
                step['function']()
            input("Press Enter to proceed to next step...")

# ----------------------------------------
# Example Usage
# ----------------------------------------

if __name__ == "__main__":
    # Generate glyph visualization for Western Witchcraft Circle
    generate_magick_circle_glyph("Western_Witchcraft")

    # Create a ritual script for the same circle
    ritual = RitualSequence("Western_Witchcraft")
    ritual.add_step(
        "Draw the protective magick circle on the ground.",
        invocation_targets=['north', 'east', 'south', 'west']
    )
    ritual.add_step(
        "Invoke the guardians to each cardinal point for protection."
    )
    ritual.add_step(
        "Charge the circle with elemental energy.",
        function=lambda: print("Playing elemental sonic mantra...")
    )
    ritual.add_step(
        "Declare intent and meditate within the circle."
    )

    ritual.run()
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
# Expanded Archetypes API Functions
# --------------------------------------------

def get_archetype(category, name):
    """
    Retrieve archetype data by category and name.
    Categories: 'planets', 'metals', 'magick_circles', 'fractal_patterns', etc.
    """
    data = globals().get(category)
    if not data:
        return None
    return data.get(name)

def list_categories():
    """
    List all archetype categories currently defined.
    """
    return [key for key in globals().keys() if isinstance(globals()[key], dict)]

def filter_archetypes_by_element(element):
    """
    Return list of archetypes matching the given element across categories.
    """
    results = []
    for category_name in list_categories():
        cat = globals().get(category_name)
        if not cat:
            continue
        for name, attrs in cat.items():
            if attrs.get('element', '').lower() == element.lower():
                results.append((category_name, name, attrs))
    return results

def filter_archetypes_by_polarity(polarity):
    """
    Return list of archetypes matching given polarity ('Yin', 'Yang', 'Balanced', etc.) across categories.
    """
    results = []
    for category_name in list_categories():
        cat = globals().get(category_name)
        if not cat:
            continue
        for name, attrs in cat.items():
            if attrs.get('polarity', '').lower() == polarity.lower():
                results.append((category_name, name, attrs))
    return results

def filter_archetypes_by_gender(gender):
    """
    Return list of archetypes matching gender ('male', 'female', 'neutral').
    """
    results = []
    for category_name in list_categories():
        cat = globals().get(category_name)
        if not cat:
            continue
        for name, attrs in cat.items():
            if attrs.get('gender', '').lower() == gender.lower():
                results.append((category_name, name, attrs))
    return results

def combined_filter(element=None, polarity=None, gender=None):
    """
    Return archetypes matching combined criteria. None means no filtering on that attribute.
    """
    def matches(attrs):
        if element and attrs.get('element', '').lower() != element.lower():
            return False
        if polarity and attrs.get('polarity', '').lower() != polarity.lower():
            return False
        if gender and attrs.get('gender', '').lower() != gender.lower():
            return False
        return True

    results = []
    for category_name in list_categories():
        cat = globals().get(category_name)
        if not cat:
            continue
        for name, attrs in cat.items():
            if matches(attrs):
                results.append((category_name, name, attrs))
    return results

def validate_archetype_data():
    """
    Validate archetypal data completeness and consistency.
    Checks for required keys: 'element', 'polarity', 'gender' and basic value sanity.
    Returns list of incomplete or inconsistent entries.
    """
    issues = []
    required_keys = ['element', 'polarity', 'gender']

    for category_name in list_categories():
        cat = globals().get(category_name)
        for name, attrs in cat.items():
            for key in required_keys:
                if key not in attrs:
                    issues.append(f"Missing '{key}' in {category_name}:{name}")
                elif not isinstance(attrs[key], str):
                    issues.append(f"Invalid type for '{key}' in {category_name}:{name}")

            # Basic polarity sanity check
            valid_pols = {'yin', 'yang', 'balanced', 'none'}
            pol = attrs.get('polarity', '').lower()
            if pol and pol not in valid_pols:
                issues.append(f"Unknown polarity '{pol}' in {category_name}:{name}")

    return issues

def cross_reference_magick_circle_to_archetypes(circle_name):
    """
    For a given magick circle, cross-reference segments to archetypes by element and polarity,
    returning detailed linked archetypal info.
    """
    from archetypes import magick_circles  # Adjust import as needed

    circle = magick_circles.get(circle_name)
    if not circle:
        return None

    detailed_segments = {}
    for seg_name, seg_attrs in circle.get('segments', {}).items():
        element = seg_attrs.get('element')
        polarity = seg_attrs.get('polarity')
        gender = seg_attrs.get('gender')
        linked = combined_filter(element=element, polarity=polarity, gender=gender)
        detailed_segments[seg_name] = {
            'segment_attributes': seg_attrs,
            'linked_archetypes': linked
        }
    return detailed_segments

def get_fractal_pattern(name):
    """
    Retrieve fractal pattern archetype data by name.
    """
    fractal_patterns = globals().get('fractal_patterns')
    if not fractal_patterns:
        return None
    return fractal_patterns.get(name)

# Example usage:
# issues = validate_archetype_data()
# fire_yang_males = combined_filter(element='Fire', polarity='Yang', gender='male')
# detailed_circle = cross_reference_magick_circle_to_archetypes('Western_Witchcraft')
# --------------------------------------------
# Section Z: Fractal Grammar Integration
# --------------------------------------------

# Polarity tolerance parameter sourced from your fractal grammar system
fractal_polarity_tolerance = 0.3

def get_fractal_grammar_rules():
    """
    Access fractal grammar rules dictionary from fractal_grammar_advanced module.
    """
    try:
        from fractal_grammar_advanced import fractal_grammar_rules
        return fractal_grammar_rules
    except ImportError:
        return {}

def get_fractal_polarity_tolerance():
    """
    Returns the polarity tolerance threshold for fractal grammar expansions.
    """
    return fractal_polarity_tolerance

def validate_fractal_grammar_terminals(archetype_symbols):
    """
    Validate that fractal grammar terminal symbols exist within archetypal symbols.
    Returns list of unknown terminals for review.
    """
    rules = get_fractal_grammar_rules()
    terminals = rules.get("terminals", {})
    unknowns = [term for term, attrs in terminals.items() if attrs.get('symbol') not in archetype_symbols]
    return unknowns

# Example (comment or remove for production):
if __name__ == "__main__":
    archetype_symbols = {attr['symbol'] for attr in globals().get('planets', {}).values()}
    unknown_symbols = validate_fractal_grammar_terminals(archetype_symbols)
    if unknown_symbols:
        print(f"Unknown fractal grammar terminals detected: {unknown_symbols}")
    else:
        print("Fractal grammar terminals validated.")
        import json

def load_archetypes(filepath='archetypes.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        archetypes = json.load(f)
    return archetypes

def archetype_by_symbol(symbol, archetypes):
    for archetype in archetypes:
        if archetype['symbol'] == symbol:
            return archetype
    return None