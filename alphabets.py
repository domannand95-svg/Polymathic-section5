"""
Polymathic Core: Alphabets Module - Hebrew Section

Includes full Hebrew alphabet letters with symbolic archetypes,
numeric gematria values, and temporal glyphs for fractal linguistics
and harmonic synthesis.

This section is designed to be extended one language at a time for
completeness and clarity.
"""

from typing import List, Dict, Optional, Union
import unicodedata

# --- Hebrew Alphabet Data ---

HEBREW_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "א", "archetype": "Aleph - Ox, Leader", "value": 1},
    {"letter": "ב", "archetype": "Bet - House, Builder", "value": 2},
    {"letter": "ג", "archetype": "Gimel - Camel, Movement", "value": 3},
    {"letter": "ד", "archetype": "Dalet - Door, Pathway", "value": 4},
    {"letter": "ה", "archetype": "Heh - Window, Revelation", "value": 5},
    {"letter": "ו", "archetype": "Vav - Hook, Connection", "value": 6},
    {"letter": "ז", "archetype": "Zayin - Sword, Protection", "value": 7},
    {"letter": "ח", "archetype": "Chet - Fence, Life", "value": 8},
    {"letter": "ט", "archetype": "Tet - Basket, Goodness", "value": 9},
    {"letter": "י", "archetype": "Yod - Hand, Divine Will", "value": 10},
    {"letter": "כ", "archetype": "Kaf - Palm, Potential", "value": 20},
    {"letter": "ל", "archetype": "Lamed - Ox Goad, Learning", "value": 30},
    {"letter": "מ", "archetype": "Mem - Water, Depth", "value": 40},
    {"letter": "נ", "archetype": "Nun - Fish, Continuity", "value": 50},
    {"letter": "ס", "archetype": "Samekh - Prop, Support", "value": 60},
    {"letter": "ע", "archetype": "Ayin - Eye, Insight", "value": 70},
    {"letter": "פ", "archetype": "Pe - Mouth, Expression", "value": 80},
    {"letter": "צ", "archetype": "Tsadi - Hook, Righteousness", "value": 90},
    {"letter": "ק", "archetype": "Qof - Back of Head, Holiness", "value": 100},
    {"letter": "ר", "archetype": "Resh - Head, Beginning", "value": 200},
    {"letter": "ש", "archetype": "Shin - Tooth, Transformation", "value": 300},
    {"letter": "ת", "archetype": "Tav - Mark, Truth", "value": 400},

    # Time-related Special Symbolic Units
    {"letter": "⏳", "archetype": "Hourglass - Time Unit", "value": 1},
    {"letter": "🕒", "archetype": "Clock Face Three O’clock - 3 Hours", "value": 3},
    {"letter": "🕦", "archetype": "Clock Face Eleven Thirty - Half Hour", "value": 0.5},
    {"letter": "⌛", "archetype": "Hourglass Done - Completion Marker", "value": 0},
]

# Language aliases for Hebrew
LANGUAGE_ALIASES = {
    "heb": "Hebrew",
    "hebrew": "Hebrew",
}

# ---------------------------
# Utility Functions
# ---------------------------

def normalize_language(lang_input: str) -> str:
    """
    Normalize language input string using aliases.
    """
    return LANGUAGE_ALIASES.get(lang_input.lower(), lang_input)

def normalize_text(text: str) -> str:
    """
    Unicode NFC normalize and strip whitespace.
    """
    return unicodedata.normalize("NFC", text).strip()

def get_alphabet(language: str) -> List[Dict]:
    """
    Return alphabet list for the normalized language key.
    """
    lang_key = normalize_language(language)
    if lang_key == "Hebrew":
        return HEBREW_ALPHABET
    return []

def lookup_letter(language: str, letter: str) -> Optional[Dict]:
    """
    Lookup letter data dict for a language.
    """
    alphabet = get_alphabet(language)
    for entry in alphabet:
        if entry["letter"] == letter:
            return entry
    return None

def translate_text(
    language: str, text: str, to_values: bool = False
) -> List[Union[str, float, None]]:
    """
    Translate a text string to a list of archetypes or numeric values.
    Unknown letters mapped to None.
    """
    norm_text = normalize_text(text)
    alphabet = get_alphabet(language)
    sorted_letters = sorted(alphabet, key=lambda e: len(e["letter"]), reverse=True)

    output = []
    i = 0
    while i < len(norm_text):
        match_found = False
        for entry in sorted_letters:
            ltr = entry["letter"]
            if norm_text[i : i + len(ltr)] == ltr:
                output.append(entry["value"] if to_values else entry["archetype"])
                i += len(ltr)
                match_found = True
                break
        if not match_found:
            output.append(None)
            i += 1
    return output

def add_language_alphabet(language: str, alphabet_list: List[Dict[str, Union[str, float]]]):
    """
    Dynamically add or extend alphabets (to be implemented later as we add languages).
    Placeholder for extension.
    """
    raise NotImplementedError("Dynamic alphabet addition not implemented yet.")

# === Self-Test ===
if __name__ == "__main__":
    sample_text = "אב⏳ד🕒ת"
    print("Input Hebrew with time glyphs:", sample_text)
    print("Archetypes:", translate_text("Hebrew", sample_text))
    print("Values:", translate_text("Hebrew", sample_text, to_values=True))
    # --- Arabic Alphabet Data ---

ARABIC_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "ا", "archetype": "Alif - Origin, Unity", "value": 1},
    {"letter": "ب", "archetype": "Ba - Door, Beginning", "value": 2},
    {"letter": "ت", "archetype": "Ta - Structure, Foundation", "value": 400},
    {"letter": "ث", "archetype": "Tha - Speech, Wisdom", "value": 500},
    {"letter": "ج", "archetype": "Jim - Water, Flow", "value": 3},
    {"letter": "ح", "archetype": "Ha - Life, Breath", "value": 8},
    {"letter": "خ", "archetype": "Kha - Protection, Barrier", "value": 600},
    {"letter": "د", "archetype": "Dal - Door, Passage", "value": 4},
    {"letter": "ذ", "archetype": "Dhal - Gift, Elevation", "value": 700},
    {"letter": "ر", "archetype": "Ra - Path, Journey", "value": 200},
    {"letter": "ز", "archetype": "Zay - Beauty, Decoration", "value": 7},
    {"letter": "س", "archetype": "Sin - Tooth, Sharpness", "value": 60},
    {"letter": "ش", "archetype": "Shin - Flame, Transformation", "value": 300},
    {"letter": "ص", "archetype": "Sad - Support, Strength", "value": 90},
    {"letter": "ض", "archetype": "Dad - Judgment, Authority", "value": 800},
    {"letter": "ط", "archetype": "Ta - Purity, Truth", "value": 9},
    {"letter": "ظ", "archetype": "Za - Light, Vision", "value": 900},
    {"letter": "ع", "archetype": "Ain - Eye, Insight", "value": 70},
    {"letter": "غ", "archetype": "Ghain - Mystery, Spirit", "value": 1000},
    {"letter": "ف", "archetype": "Fa - Opening, Opportunity", "value": 80},
    {"letter": "ق", "archetype": "Qaf - Heart, Depth", "value": 100},
    {"letter": "ك", "archetype": "Kaf - Palm, Receiving", "value": 20},
    {"letter": "ل", "archetype": "Lam - Ox Goad, Direction", "value": 30},
    {"letter": "م", "archetype": "Mim - Water, Flow", "value": 40},
    {"letter": "ن", "archetype": "Nun - Seed, Continuity", "value": 50},
    {"letter": "ه", "archetype": "Ha - Breath, Spirit", "value": 5},
    {"letter": "و", "archetype": "Waw - Hook, Connection", "value": 6},
    {"letter": "ي", "archetype": "Ya - Hand, Guidance", "value": 10},

    # Time-related special or symbolic units
    {"letter": "⏰", "archetype": "Alarm Clock - Time Marker", "value": 1},
    {"letter": "🕰️", "archetype": "Mantelpiece Clock - Hour Unit", "value": 1},
    {"letter": "⌚", "archetype": "Wristwatch - Small Time Unit", "value": 0.1},
    {"letter": "🕑", "archetype": "Clock Face Two O’clock - Two Hours", "value": 2},
]# --- Latin Alphabet Data ---

LATIN_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "A", "archetype": "Alpha - Initiator, Leader", "value": 1},
    {"letter": "B", "archetype": "Builder, Structure", "value": 2},
    {"letter": "C", "archetype": "Flow, Movement", "value": 3},
    {"letter": "D", "archetype": "Door, Gateway", "value": 4},
    {"letter": "E", "archetype": "Messenger, Energy", "value": 5},
    {"letter": "F", "archetype": "Force, Power", "value": 6},
    {"letter": "G", "archetype": "Growth, Expansion", "value": 7},
    {"letter": "H", "archetype": "Foundation, Breath", "value": 8},
    {"letter": "I", "archetype": "Individual, Seed", "value": 9},
    {"letter": "J", "archetype": "Journey, Transformation", "value": 10},
    {"letter": "K", "archetype": "Key, Action", "value": 11},
    {"letter": "L", "archetype": "Light, Learning", "value": 12},
    {"letter": "M", "archetype": "Mother, Water", "value": 13},
    {"letter": "N", "archetype": "Nature, Continuity", "value": 14},
    {"letter": "O", "archetype": "Origin, Unity", "value": 15},
    {"letter": "P", "archetype": "Power, Expression", "value": 16},
    {"letter": "Q", "archetype": "Question, Mystery", "value": 17},
    {"letter": "R", "archetype": "Root, Strength", "value": 18},
    {"letter": "S", "archetype": "Sun, Success", "value": 19},
    {"letter": "T", "archetype": "Truth, Stability", "value": 20},
    {"letter": "U", "archetype": "Unity, Union", "value": 21},
    {"letter": "V", "archetype": "Victory, Life Force", "value": 22},
    {"letter": "W", "archetype": "Wave, Change", "value": 23},
    {"letter": "X", "archetype": "Cross, Intersection", "value": 24},
    {"letter": "Y", "archetype": "Path, Question", "value": 25},
    {"letter": "Z", "archetype": "End, Completion", "value": 26},

    # Time-related symbolic units
    {"letter": "⏰", "archetype": "Alarm Clock - Time Marker", "value": 1},
    {"letter": "⌚", "archetype": "Wristwatch - Small Time Unit", "value": 0.1},
    {"letter": "🕰️", "archetype": "Mantelpiece Clock - Hour Unit", "value": 1},
    {"letter": "🕒", "archetype": "Clock Face Three O’clock - 3 Hours", "value": 3},
]ALPHABETS.update({
    "Latin": LATIN_ALPHABET,
    # other lang entries...
})# --- Greek Alphabet Data ---

GREEK_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "Α", "archetype": "Alpha - Beginning, Leader", "value": 1},
    {"letter": "Β", "archetype": "Beta - House, Foundation", "value": 2},
    {"letter": "Γ", "archetype": "Gamma - Earth, Strength", "value": 3},
    {"letter": "Δ", "archetype": "Delta - Door, Change", "value": 4},
    {"letter": "Ε", "archetype": "Epsilon - Life, Energy", "value": 5},
    {"letter": "Ζ", "archetype": "Zeta - Weapon, Power", "value": 7},
    {"letter": "Η", "archetype": "Eta - Sun, Spirit", "value": 8},
    {"letter": "Θ", "archetype": "Theta - Death, Spirit", "value": 9},
    {"letter": "Ι", "archetype": "Iota - Small, Seed", "value": 10},
    {"letter": "Κ", "archetype": "Kappa - Hand, Action", "value": 20},
    {"letter": "Λ", "archetype": "Lambda - Wolf, Leadership", "value": 30},
    {"letter": "Μ", "archetype": "Mu - Water, Flow", "value": 40},
    {"letter": "Ν", "archetype": "Nu - Fish, Life", "value": 50},
    {"letter": "Ξ", "archetype": "Xi - Wave, Change", "value": 60},
    {"letter": "Ο", "archetype": "Omicron - Eye, Perception", "value": 70},
    {"letter": "Π", "archetype": "Pi - Door, Transition", "value": 80},
    {"letter": "Ρ", "archetype": "Rho - Head, Authority", "value": 100},
    {"letter": "Σ", "archetype": "Sigma - Sun, Victory", "value": 200},
    {"letter": "Τ", "archetype": "Tau - Mark, Cross", "value": 300},
    {"letter": "Υ", "archetype": "Upsilon - Aspiration, Spirit", "value": 400},
    {"letter": "Φ", "archetype": "Phi - Nature, Growth", "value": 500},
    {"letter": "Χ", "archetype": "Chi - Life, Spirit", "value": 600},
    {"letter": "Ψ", "archetype": "Psi - Soul, Depth", "value": 700},
    {"letter": "Ω", "archetype": "Omega - End, Completion", "value": 800},

    # Time-related symbolic units
    {"letter": "⏱️", "archetype": "Stopwatch - Precise Time Unit", "value": 0.1},
    {"letter": "⏰", "archetype": "Alarm Clock - Time Marker", "value": 1},
    {"letter": "🕰️", "archetype": "Mantelpiece Clock - Hour Unit", "value": 1},
    {"letter": "🕙", "archetype": "Clock Face Ten O’clock - Ten Hours", "value": 10},
]ALPHABETS.update({
    "Greek": GREEK_ALPHABET,
    # other alphabets...
})# --- Chinese Radicals & Numerals Data ---

CHINESE_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "一", "archetype": "One, Unity, Line", "value": 1},
    {"letter": "丨", "archetype": "Line, Connection", "value": 2},
    {"letter": "丶", "archetype": "Dot, Point", "value": 3},
    {"letter": "丿", "archetype": "Slash, Movement", "value": 4},
    {"letter": "乙", "archetype": "Second, Turning", "value": 5},
    {"letter": "亅", "archetype": "Hook, Grip", "value": 6},
    {"letter": "口", "archetype": "Mouth, Speech, Expression", "value": 7},
    {"letter": "女", "archetype": "Woman, Feminine", "value": 8},
    {"letter": "手", "archetype": "Hand, Action", "value": 9},
    {"letter": "水", "archetype": "Water, Flow", "value": 10},
    {"letter": "火", "archetype": "Fire, Energy", "value": 11},
    {"letter": "木", "archetype": "Tree, Growth", "value": 12},
    {"letter": "金", "archetype": "Metal, Strength", "value": 13},
    {"letter": "土", "archetype": "Earth, Foundation", "value": 14},
    {"letter": "日", "archetype": "Sun, Light", "value": 15},
    {"letter": "月", "archetype": "Moon, Cycle", "value": 16},
    {"letter": "山", "archetype": "Mountain, Stability", "value": 17},
    {"letter": "田", "archetype": "Field, Fertility", "value": 18},
    {"letter": "目", "archetype": "Eye, Vision", "value": 19},
    {"letter": "禾", "archetype": "Grain, Nourishment", "value": 20},
    {"letter": "言", "archetype": "Speech, Communication", "value": 21},
    {"letter": "足", "archetype": "Foot, Movement", "value": 22},
    {"letter": "贝", "archetype": "Shell (Money), Wealth", "value": 23},

    # Chinese numerals 0-9
    {"letter": "零", "archetype": "Zero, Nothingness", "value": 0},
    {"letter": "一", "archetype": "One, Unity", "value": 1},
    {"letter": "二", "archetype": "Two, Duality", "value": 2},
    {"letter": "三", "archetype": "Three, Trinity", "value": 3},
    {"letter": "四", "archetype": "Four, Stability", "value": 4},
    {"letter": "五", "archetype": "Five, Balance", "value": 5},
    {"letter": "六", "archetype": "Six, Harmony", "value": 6},
    {"letter": "七", "archetype": "Seven, Mystery", "value": 7},
    {"letter": "八", "archetype": "Eight, Prosperity", "value": 8},
    {"letter": "九", "archetype": "Nine, Completion", "value": 9},

    # Time-related symbols
    {"letter": "时", "archetype": "Hour, Time Unit", "value": 1},
    {"letter": "分", "archetype": "Minute, Division of Time", "value": 0.0167},  # approx 1/60 hour
    {"letter": "秒", "archetype": "Second, Time Measure", "value": 0.0002778},  # approx 1/3600 hour
    {"letter": "晨", "archetype": "Morning, Start of Day", "value": 0},
    {"letter": "夜", "archetype": "Night, Darkness Cycle", "value": 0},
]ALPHABETS.update({
    "Chinese": CHINESE_ALPHABET,
    # other alphabets...
})# --- Egyptian Hieroglyphs Alphabet Data ---

EGYPTIAN_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "𓄿", "archetype": "A - Vulture, Spirit, Beginning", "value": 1},
    {"letter": "𓃀", "archetype": "B - Foot, Movement, Foundation", "value": 2},
    {"letter": "𓂧", "archetype": "D - Hand, Power, Action", "value": 3},
    {"letter": "𓆓", "archetype": "F - Horned Viper, Force, Danger", "value": 4},
    {"letter": "𓎼", "archetype": "G - Stand, Strength, Growth", "value": 5},
    {"letter": "𓉔", "archetype": "H - Twist of Flax, Breath, Spirit", "value": 6},
    {"letter": "𓇋", "archetype": "I/Y - Reed, Life, Growth", "value": 7},
    {"letter": "𓎡", "archetype": "K - Hill, Stability, Earth", "value": 8},
    {"letter": "𓂓", "archetype": "L - Lion, Strength, Leadership", "value": 9},
    {"letter": "𓈖", "archetype": "N - Water, Flow, Life", "value": 10},
    {"letter": "𓊪", "archetype": "P - Stool, Seat, Authority", "value": 20},
    {"letter": "𓂋", "archetype": "R - Mouth, Speech, Expression", "value": 30},
    {"letter": "𓈙", "archetype": "Sh - Pool, Water, Depth", "value": 40},
    {"letter": "𓐍", "archetype": "Kh - Placenta, Life, Birth", "value": 50},
    {"letter": "𓅱", "archetype": "W - Quail Chick, Spirit, Smallness", "value": 60},

    # Egyptian numeral glyphs
    {"letter": "𓏺", "archetype": "Stroke, Unit One", "value": 1},
    {"letter": "𓎆", "archetype": "Heel Bone, Ten", "value": 10},
    {"letter": "𓍢", "archetype": "Coil of Rope, Hundred", "value": 100},
    {"letter": "𓆼", "archetype": "Lotus Flower, Thousand", "value": 1000},
    {"letter": "𓂭", "archetype": "Finger, Ten Thousand", "value": 10000},
    {"letter": "𓆐", "archetype": "Tadpole/Frog, Hundred Thousand", "value": 100000},
    {"letter": "𓁨", "archetype": "Astonished Man, Million", "value": 1000000},

    # Time-related glyphs (hour, smaller/larger units)
    {"letter": "𓏲", "archetype": "Hourglass, Hour Unit", "value": 1},
    {"letter": "𓎲", "archetype": "Alternate Hour Glyph", "value": 1},
    {"letter": "𓇽", "archetype": "Minute Symbol (example)", "value": 1/60},
    {"letter": "𓏤", "archetype": "Day Sign, Solar Cycle", "value": 24},
    {"letter": "𓇹", "archetype": "Night/Darkness Cycle", "value": 24},
]ALPHABETS.update({
    "Egyptian": EGYPTIAN_ALPHABET,
    # other alphabets...
})# --- Sanskrit Alphabet Data ---

SANSKRIT_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "अ", "archetype": "Primal Sound, Creation", "value": 1},
    {"letter": "आ", "archetype": "Extension, Expansion", "value": 2},
    {"letter": "इ", "archetype": "Light, Illumination", "value": 3},
    {"letter": "ई", "archetype": "Growth, Strength", "value": 4},
    {"letter": "उ", "archetype": "Energy, Breath", "value": 5},
    {"letter": "ऊ", "archetype": "Power, Depth", "value": 6},
    {"letter": "ऋ", "archetype": "Life Force, Spirit", "value": 7},
    {"letter": "ॠ", "archetype": "Higher Consciousness", "value": 8},
    {"letter": "ऌ", "archetype": "Mysticism, Mystery", "value": 9},
    {"letter": "ए", "archetype": "Revelation, Vision", "value": 10},
    {"letter": "ऐ", "archetype": "Creative Power", "value": 11},
    {"letter": "ओ", "archetype": "Perfection, Wholeness", "value": 12},
    {"letter": "औ", "archetype": "Cosmic Breath", "value": 13},
    {"letter": "अं", "archetype": "Seed, Potential", "value": 14},
    {"letter": "अः", "archetype": "Transformation, Release", "value": 15},

    # Consonants (selected subset)
    {"letter": "क", "archetype": "Power, Action", "value": 16},
    {"letter": "ख", "archetype": "Energy, Movement", "value": 17},
    {"letter": "ग", "archetype": "Earth, Stability", "value": 18},
    {"letter": "घ", "archetype": "Air, Expansion", "value": 19},
    {"letter": "ङ", "archetype": "Root, Foundation", "value": 20},

    # Time-related symbols (traditional and symbolic)
    {"letter": "काल", "archetype": "Time, Eternal Cycle", "value": 1},
    {"letter": "क्षण", "archetype": "Moment, Instant", "value": 0.0002778},  # ~second equivalent
    {"letter": "निमेष", "archetype": "Blink, Small Time Unit", "value": 0.0001},
    {"letter": "अर्द्ध", "archetype": "Half Unit, Division", "value": 0.5},
]ALPHABETS.update({
    "Sanskrit": SANSKRIT_ALPHABET,
    # other alphabets...
})# --- Gaian Alphabet Data ---

GAIAN_ALPHABET: List[Dict[str, Union[str, float]]] = [
    {"letter": "𝔊", "archetype": "Spirit, Essence", "value": 1},
    {"letter": "𝔄", "archetype": "Life, Vitality", "value": 2},
    {"letter": "𝔦", "archetype": "Change, Adaptation", "value": 3},
    {"letter": "𝔞", "archetype": "Path, Journey", "value": 4},
    {"letter": "𝔫", "archetype": "Balance, Harmony", "value": 5},
    {"letter": "𝔟", "archetype": "Foundation, Stability", "value": 6},
    {"letter": "𝔘", "archetype": "Unity, Oneness", "value": 7},
    {"letter": "𝔩", "archetype": "Light, Illumination", "value": 8},
    {"letter": "𝔢", "archetype": "Energy, Flow", "value": 9},
    # Extend with additional letters as finalized
]ALPHABETS.update({
    "Gaian": GAIAN_ALPHABET,
    # other alphabets...
})"""
polymathic_core.alphabets

Comprehensive multilingual alphabet module.
Languages included so far:
- Hebrew
- Greek

More languages will be appended one at a time.
"""

from typing import Dict, Optional

# Hebrew alphabet section (already provided)
HEBREW_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "א": {"name": "Aleph", "archetype": "Ox, Leader, Breath, Unity", "gematria": 1, "time_glyph": "Sunrise"},
    "ב": {"name": "Bet", "archetype": "House, Creation, Duality", "gematria": 2, "time_glyph": "Morning"},
    "ג": {"name": "Gimel", "archetype": "Camel, Reward, Movement", "gematria": 3, "time_glyph": "Midday"},
    # ... full alphabet continued
    "ת": {"name": "Tav", "archetype": "Mark, Covenant, Seal", "gematria": 400, "time_glyph": "Pre-Dawn"},
}

# Greek alphabet section added
GREEK_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "Α": {"name": "Alpha", "archetype": "Beginning, Leader, Light", "numeric_value": 1},
    "Β": {"name": "Beta", "archetype": "House, Foundation, Duality", "numeric_value": 2},
    "Γ": {"name": "Gamma", "archetype": "Door, Change, Transition", "numeric_value": 3},
    "Δ": {"name": "Delta", "archetype": "Triangle, Stability, Change", "numeric_value": 4},
    "Ε": {"name": "Epsilon", "archetype": "Window, Breath, Expression", "numeric_value": 5},
    "Ζ": {"name": "Zeta", "archetype": "Weapon, Struggle, Energy", "numeric_value": 7},
    "Η": {"name": "Eta", "archetype": "Energy, Strength, Harmony", "numeric_value": 8},
    "Θ": {"name": "Theta", "archetype": "Life, Death, Protection", "numeric_value": 9},
    "Ι": {"name": "Iota", "archetype": "Point, Hand, Power", "numeric_value": 10},
    "Κ": {"name": "Kappa", "archetype": "Palm, Potential, Holding", "numeric_value": 20},
    "Λ": {"name": "Lambda", "archetype": "Path, Guide, Learning", "numeric_value": 30},
    "Μ": {"name": "Mu", "archetype": "Water, Flow, Chaos", "numeric_value": 40},
    "Ν": {"name": "Nu", "archetype": "Life, Fish, Fertility", "numeric_value": 50},
    "Ξ": {"name": "Xi", "archetype": "Struggle, Obstacle", "numeric_value": 60},
    "Ο": {"name": "Omicron", "archetype": "Circle, Cycle, Completion", "numeric_value": 70},
    "Π": {"name": "Pi", "archetype": "Mouth, Expression, Boundary", "numeric_value": 80},
    "Ρ": {"name": "Rho", "archetype": "Head, Leader, Beginning", "numeric_value": 100},
    "Σ": {"name": "Sigma", "archetype": "Sum, Change, Transformation", "numeric_value": 200},
    "Τ": {"name": "Tau", "archetype": "Mark, Cross, Covenant", "numeric_value": 300},
    "Υ": {"name": "Upsilon", "archetype": "Branch, Choice, Voice", "numeric_value": 400},
    "Φ": {"name": "Phi", "archetype": "Nature, Breath, Flower", "numeric_value": 500},
    "Χ": {"name": "Chi", "archetype": "Christ, Life, Spirit", "numeric_value": 600},
    "Ψ": {"name": "Psi", "archetype": "Soul, Spirit, Mind", "numeric_value": 700},
    "Ω": {"name": "Omega", "archetype": "End, Completion, Truth", "numeric_value": 800},
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter in the specified alphabet.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    return None


def get_letter_numeric_value(alphabet: str, letter: str) -> Optional[int]:
    """
    Retrieves the numeric or gematria value of a given letter in the specified alphabet.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("gematria")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("numeric_value")
    return None


# Simple self-test example
if __name__ == "__main__":
    letters = [("hebrew", "ג"), ("greek", "Δ")]
    for alphabet, letter in letters:
        print(f"Alphabet: {alphabet}, Letter: {letter}")
        print(f" Archetype: {get_letter_archetype(alphabet, letter)}")
        print(f" Numeric Value: {get_letter_numeric_value(alphabet, letter)}")
        print()
        """
polymathic_core.alphabets

Comprehensive multilingual alphabet module.
Languages included so far:
- Hebrew
- Greek
- Latin

Additional alphabets to be appended progressively.
"""

from typing import Dict, Optional

# Hebrew alphabet (previously included)
HEBREW_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "א": {"name": "Aleph", "archetype": "Ox, Leader, Breath, Unity", "gematria": 1, "time_glyph": "Sunrise"},
    # ... (rest of Hebrew letters)
    "ת": {"name": "Tav", "archetype": "Mark, Covenant, Seal", "gematria": 400, "time_glyph": "Pre-Dawn"},
}

# Greek alphabet (previously included)
GREEK_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "Α": {"name": "Alpha", "archetype": "Beginning, Leader, Light", "numeric_value": 1},
    # ... (rest of Greek letters)
    "Ω": {"name": "Omega", "archetype": "End, Completion, Truth", "numeric_value": 800},
}

# Latin alphabet added
LATIN_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "A": {"name": "A", "archetype": "Beginning, Breath, Source"},
    "B": {"name": "B", "archetype": "Foundation, Duality, House"},
    "C": {"name": "C", "archetype": "Curve, Change, Crossing"},
    "D": {"name": "D", "archetype": "Door, Gateway, Passage"},
    "E": {"name": "E", "archetype": "Breath, Expression, Energy"},
    "F": {"name": "F", "archetype": "Hook, Connection, Protection"},
    "G": {"name": "G", "archetype": "Earth, Gift, Container"},
    "H": {"name": "H", "archetype": "Fence, Enclosure, Life"},
    "I": {"name": "I", "archetype": "Point, Power, Hand"},
    "J": {"name": "J", "archetype": "Transformation, Hook, Flow"},
    "K": {"name": "K", "archetype": "Palm, Potential, Action"},
    "L": {"name": "L", "archetype": "Ox Goad, Learning, Guidance"},
    "M": {"name": "M", "archetype": "Water, Flow, Matrix"},
    "N": {"name": "N", "archetype": "Fish, Life, Fertility"},
    "O": {"name": "O", "archetype": "Circle, Completion"},
    "P": {"name": "P", "archetype": "Mouth, Expression"},
    "Q": {"name": "Q", "archetype": "Power, Control"},
    "R": {"name": "R", "archetype": "Head, Leader"},
    "S": {"name": "S", "archetype": "Serpent, Change"},
    "T": {"name": "T", "archetype": "Cross, Mark, Covenant"},
    "U": {"name": "U", "archetype": "Vessel, Protection"},
    "V": {"name": "V", "archetype": "Path, Choice, Victory"},
    "W": {"name": "W", "archetype": "Wave, Motion"},
    "X": {"name": "X", "archetype": "Crossroads, Intersection"},
    "Y": {"name": "Y", "archetype": "Branch, Decision"},
    "Z": {"name": "Z", "archetype": "Life, Spirit"},
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter in the specified alphabet.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    return None


def get_letter_numeric_value(alphabet: str, letter: str) -> Optional[int]:
    """
    Retrieves the numeric or gematria value of a given letter in the specified alphabet.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("gematria")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("numeric_value")
    # Latin letters do not have numeric values by default
    return None


# Simple test usage
if __name__ == "__main__":
    tests = [("hebrew", "א"), ("greek", "Δ"), ("latin", "C")]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter '{letter}': Archetype={get_letter_archetype(alphabet, letter)}, Numeric={get_letter_numeric_value(alphabet, letter)}")
        # Arabic alphabet added
ARABIC_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "ا": {"name": "Alif", "archetype": "Unity, Breath, Beginning"},
    "ب": {"name": "Ba", "archetype": "House, Creation, Receptivity"},
    "ت": {"name": "Ta", "archetype": "Mark, Sign, Feminine Energy"},
    "ث": {"name": "Tha", "archetype": "Three, Divine Attributes"},
    "ج": {"name": "Jim", "archetype": "Camel, Reward, Movement"},
    "ح": {"name": "Ha", "archetype": "Fence, Life, Protection"},
    "خ": {"name": "Kha", "archetype": "Light, Inner Fire, Secret"},
    "د": {"name": "Dal", "archetype": "Door, Pathway"},
    "ذ": {"name": "Dhal", "archetype": "Arc, Extension"},
    "ر": {"name": "Ra", "archetype": "Head, Leader, Flow"},
    "ز": {"name": "Zay", "archetype": "Weapon, Struggle"},
    "س": {"name": "Sin", "archetype": "Teeth, Biting, Change"},
    "ش": {"name": "Shin", "archetype": "Fire, Tooth, Spirit"},
    "ص": {"name": "Sad", "archetype": "Hunting, Righteousness"},
    "ض": {"name": "Dad", "archetype": "Echo, Creation, Power"},
    "ط": {"name": "Ta", "archetype": "Mark, Sign"},
    "ظ": {"name": "Dha", "archetype": "Radiance, Judgment"},
    "ع": {"name": "Ain", "archetype": "Eye, Insight, Perception"},
    "غ": {"name": "Ghain", "archetype": "Rain, Renewal"},
    "ف": {"name": "Fa", "archetype": "Mouth, Speech"},
    "ق": {"name": "Qaf", "archetype": "Back of Head, Spirituality"},
    "ك": {"name": "Kaf", "archetype": "Palm, Power"},
    "ل": {"name": "Lam", "archetype": "Ox Goad, Learning"},
    "م": {"name": "Mim", "archetype": "Water, Flow, Mystery"},
    "ن": {"name": "Nun", "archetype": "Fish, Life"},
    "ه": {"name": "Ha", "archetype": "Breath, Revelation"},
    "و": {"name": "Waw", "archetype": "Hook, Connection"},
    "ي": {"name": "Ya", "archetype": "Hand, Divine Will"},
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter in the specified alphabet.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    return None

# Example usage test
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        # Chinese characters (selected core set for archetypal encoding)
CHINESE_CHARACTERS: Dict[str, Dict[str, Optional[str]]] = {
    "一": {"name": "One", "archetype": "Unity, Beginning, Oneness"},
    "二": {"name": "Two", "archetype": "Duality, Balance, Polarity"},
    "三": {"name": "Three", "archetype": "Harmony, Growth, Creation"},
    "水": {"name": "Water", "archetype": "Flow, Emotion, Adaptability"},
    "火": {"name": "Fire", "archetype": "Energy, Transformation, Passion"},
    "木": {"name": "Wood", "archetype": "Life, Growth, Strength"},
    "金": {"name": "Metal", "archetype": "Purity, Rigidity, Precision"},
    "土": {"name": "Earth", "archetype": "Stability, Nourishment, Grounding"},
    "风": {"name": "Wind", "archetype": "Movement, Change, Freedom"},
    "山": {"name": "Mountain", "archetype": "Strength, Stillness, Endurance"},
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter/character in the specified alphabet/script.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "chinese":
        return CHINESE_CHARACTERS.get(letter, {}).get("archetype")
    return None

# Example usage test including Chinese
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
        ("chinese", "水"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter/Character '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        # Egyptian transliteration alphabet section
EGYPTIAN_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "𓀀": {"name": "A", "archetype": "Vulture, Divine Feminine, Creator"},
    "𓃾": {"name": "B", "archetype": "Foot, Movement, Foundation"},
    "𓎡": {"name": "K", "archetype": "Basket, Container, Power"},
    "𓂓": {"name": "Q", "archetype": "Hill, Strength, Endurance"},
    "𓂝": {"name": "Aʾ", "archetype": "Arm, Action, Creation"},
    "𓈎": {"name": "Ch", "archetype": "Rope, Binding, Connection"},
    "𓅱": {"name": "W", "archetype": "Quail Chick, Sound, Communication"},
    "𓊃": {"name": "S", "archetype": "Folded Cloth, Protection"},
    "𓌳": {"name": "M", "archetype": "Owl, Wisdom, Mystery"},
    "𓏏": {"name": "T", "archetype": "Bread, Offering, Life"},
    # Extend as needed for all common transliteration glyphs
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter/character in the specified alphabet/script.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "chinese":
        return CHINESE_CHARACTERS.get(letter, {}).get("archetype")
    elif alphabet == "egyptian":
        return EGYPTIAN_ALPHABET.get(letter, {}).get("archetype")
    return None

# Example usage test including Egyptian transliteration
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
        ("chinese", "水"),
        ("egyptian", "𓏏"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter/Character '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        # Enochian alphabet section
ENOCHIAN_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "Pa": {"name": "Pa", "archetype": "Power, Door, Beginning"},
    "Veh": {"name": "Veh", "archetype": "Vision, Revelation"},
    "Don": {"name": "Don", "archetype": "Gift, Grace"},
    "Ged": {"name": "Ged", "archetype": "Knowledge, Wisdom"},
    "Gal": {"name": "Gal", "archetype": "Force, Strength"},
    "Gis": {"name": "Gis", "archetype": "Light, Illumination"},
    "Tal": {"name": "Tal", "archetype": "Motion, Change"},
    "Med": {"name": "Med", "archetype": "Foundation, Stability"},
    "Gah": {"name": "Gah", "archetype": "Fire, Spirit"},
    "Na": {"name": "Na", "archetype": "Water, Emotion"},
    # Extend full Enochian alphabet according to tradition...
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter/character in the specified alphabet/script.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "chinese":
        return CHINESE_CHARACTERS.get(letter, {}).get("archetype")
    elif alphabet == "egyptian":
        return EGYPTIAN_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "enochian":
        return ENOCHIAN_ALPHABET.get(letter, {}).get("archetype")
    return None

# Example usage test including Enochian
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
        ("chinese", "水"),
        ("egyptian", "𓏏"),
        ("enochian", "Pa"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter/Character '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        # Sanskrit Devanagari alphabet section
SANSKRIT_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "अ": {"name": "A", "archetype": "Beginning, Source, Breath"},
    "आ": {"name": "Ā", "archetype": "Expansion, Warmth"},
    "इ": {"name": "I", "archetype": "Light, Insight"},
    "ई": {"name": "Ī", "archetype": "Intensity, Illumination"},
    "उ": {"name": "U", "archetype": "Flow, Water"},
    "ऊ": {"name": "Ū", "archetype": "Depth, Strength"},
    "ऋ": {"name": "ṛ", "archetype": "Life Force, Vitality"},
    "ए": {"name": "E", "archetype": "Energy, Awareness"},
    "ऐ": {"name": "Ai", "archetype": "Expansion, Creativity"},
    "ओ": {"name": "O", "archetype": "Completion, Unity"},
    "औ": {"name": "Au", "archetype": "Transition, Power"},
    "क": {"name": "Ka", "archetype": "Earth, Foundation"},
    "ख": {"name": "Kha", "archetype": "Air, Movement"},
    "ग": {"name": "Ga", "archetype": "Fire, Energy"},
    "घ": {"name": "Gha", "archetype": "Spirit, Transformation"},
    "च": {"name": "Ca", "archetype": "Water, Flow"},
    "छ": {"name": "Cha", "archetype": "Breath, Expansion"},
    "ज": {"name": "Ja", "archetype": "Creation, Manifestation"},
    # Extend full Devanagari alphabet
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    """
    Retrieves the archetype description of a given letter/character in the specified alphabet/script.
    """
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "chinese":
        return CHINESE_CHARACTERS.get(letter, {}).get("archetype")
    elif alphabet == "egyptian":
        return EGYPTIAN_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "enochian":
        return ENOCHIAN_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "sanskrit":
        return SANSKRIT_ALPHABET.get(letter, {}).get("archetype")
    return None

# Example usage test including Sanskrit
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
        ("chinese", "水"),
        ("egyptian", "𓏏"),
        ("enochian", "Pa"),
        ("sanskrit", "अ"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter/Character '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        # Gaian Sanskrit alphabet section (symbolic variant of Devanagari)
GAIAN_SANSKRIT_ALPHABET: Dict[str, Dict[str, Optional[str]]] = {
    "𑀅": {"name": "A", "archetype": "Universal Source, Breath, Unity"},
    "𑀆": {"name": "Ā", "archetype": "Expansion, Cosmic Warmth"},
    "𑀇": {"name": "I", "archetype": "Light, Inner Sight"},
    "𑀈": {"name": "Ī", "archetype": "Intensity, Illumination"},
    "𑀉": {"name": "U", "archetype": "Flow, Life Energy"},
    "𑀊": {"name": "Ū", "archetype": "Depth, Strength, Vitality"},
    "𑀋": {"name": "ṛ", "archetype": "Life Force, Universal Pulse"},
    "𑀏": {"name": "E", "archetype": "Energy, Awareness Expansion"},
    "𑀐": {"name": "Ai", "archetype": "Creative Growth, Transformation"},
    "𑀑": {"name": "O", "archetype": "Completion, Oneness"},
    "𑀒": {"name": "Au", "archetype": "Transition, Cosmic Power"},
    "𑀓": {"name": "Ka", "archetype": "Foundation, Earth Element"},
    "𑀔": {"name": "Kha", "archetype": "Air, Breath, Movement"},
    "𑀕": {"name": "Ga", "archetype": "Fire, Transformation"},
    "𑀖": {"name": "Gha", "archetype": "Spirit, Illumination"},
    # Extend as needed with remaining Gaian Sanskrit glyphs
}

def get_letter_archetype(alphabet: str, letter: str) -> Optional[str]:
    alphabet = alphabet.lower()
    if alphabet == "hebrew":
        return HEBREW_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "greek":
        return GREEK_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "latin":
        return LATIN_ALPHABET.get(letter.upper(), {}).get("archetype")
    elif alphabet == "arabic":
        return ARABIC_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "chinese":
        return CHINESE_CHARACTERS.get(letter, {}).get("archetype")
    elif alphabet == "egyptian":
        return EGYPTIAN_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "enochian":
        return ENOCHIAN_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "sanskrit":
        return SANSKRIT_ALPHABET.get(letter, {}).get("archetype")
    elif alphabet == "gaian_sanskrit":
        return GAIAN_SANSKRIT_ALPHABET.get(letter, {}).get("archetype")
    return None

# Example usage test including Gaian Sanskrit
if __name__ == "__main__":
    tests = [
        ("hebrew", "א"),
        ("greek", "Δ"),
        ("latin", "C"),
        ("arabic", "ج"),
        ("chinese", "水"),
        ("egyptian", "𓏏"),
        ("enochian", "Pa"),
        ("sanskrit", "अ"),
        ("gaian_sanskrit", "𑀅"),
    ]
    for alphabet, letter in tests:
        print(f"{alphabet.title()} Letter/Character '{letter}': Archetype={get_letter_archetype(alphabet, letter)}")
        