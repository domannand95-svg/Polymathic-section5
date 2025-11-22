"""
Unified Symbolic Dataset and Numeric Extraction Utilities with Gender/Polarity Multipliers.

Supports: Hebrew, Greek, Sanskrit, Arabic, Enochian, Chinese, Egyptian

Provides:
- Symbol-to-numeric value extraction per traditional numerologies.
- Gender/polarity multipliers (+1/-1) per symbol integrated in extraction.
- Functions returning both base and polarized numeric values.
"""

# Alphabets with letters/symbols and numeric values
alphabets = {
    'Hebrew': {
        'letters': 'אבגדהוזחטיכלמנסעפצקרשת',
        'values': [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400],
    },
    'Greek': {
        'letters': 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ',
        'values': [1,2,3,4,5,7,8,9,10,20,30,40,50,60,70,80,100,200,300,400,500,600,700,800],
    },
    'Sanskrit': {
        'letters': ['ka','kha','ga','gha','ṅa','ca','cha','ja','jha','ña',
                    'ṭa','ṭha','ḍa','ḍha','ṇa','ta','tha','da','dha','na',
                    'pa','pha','ba','bha','ma','ya','ra','la','va','śa','ṣa','sa','ha'],
        'values': [1,2,3,4,5,6,7,8,9,0,
                   1,2,3,4,5,6,7,8,9,0,
                   1,2,3,4,5,6,7,8,9,0,0,0,0]
    },
    'Arabic': {
        'letters': 'ابجد هوز حطي كلمن سعفص قرشت ثخذ ضظغ'.replace(' ',''),
        'values': [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000],
    },
    'Enochian': {
        'letters': ['Un','Pa','Veh','Gal','Gere','Madah','Pol','Gon','Tol','Des',
                    'Len','I','Fam','Ur','Med','Na','Bit','Cal','Bis','Dos','Mit'],
    },
    'Chinese': {
        'letters': ['甲','乙','丙','丁','戊','己','庚','辛','壬','癸'],
        'values': list(range(1,11)),
    },
    'Egyptian': {
        'letters': ['Ankh', 'Djed', 'Was'],
        'values': [1, 2, 3],
    }
}

# Gender and polarity multipliers, alternating +1/-1 by index for demonstration.
# Customize as needed per cultural/archetypal assignments.
gender_polarity_map = {
    lang: {c: (+1 if i % 2 == 0 else -1) for i, c in enumerate(data['letters'])}
    for lang, data in alphabets.items()
}

def get_numeric_value(symbol: str, alphabet: str) -> int:
    """
    Returns the base numeric value of a symbol given its alphabet.
    For Enochian (no predefined numeric), returns index + 1.
    Returns 0 if symbol/alphabet not found.
    """
    a = alphabets.get(alphabet)
    if not a:
        return 0
    if alphabet == 'Enochian':
        try:
            return a['letters'].index(symbol) + 1
        except ValueError:
            return 0
    letter_map = dict(zip(a['letters'], a['values']))
    return letter_map.get(symbol, 0)

def get_polarized_numeric_value(symbol: str, alphabet: str) -> int:
    """
    Returns the polarized numeric value, i.e., base numeric value
    multiplied by the gender/polarity multiplier for the symbol.
    Defaults multiplier to +1 if not found.
    """
    base_val = get_numeric_value(symbol, alphabet)
    polarity = gender_polarity_map.get(alphabet, {}).get(symbol, +1)
    return base_val * polarity

def batch_get_polarized_numeric_values(symbols: list, alphabet: str) -> list:
    """
    Batch processing utility: Returns a list of polarized numeric values,
    for a list of symbols all from the same alphabet.
    """
    return [get_polarized_numeric_value(sym, alphabet) for sym in symbols]

# ----- Example usage -----
if __name__ == '__main__':
    test_alphabet = 'Hebrew'
    test_symbols = ['א', 'ב', 'ג', 'ד', 'ה']
    
    # Base numeric values
    base_vals = [get_numeric_value(s, test_alphabet) for s in test_symbols]
    print(f"Base numeric values for {test_symbols} in {test_alphabet}: {base_vals}")
    
    # Polarized numeric values
    polarized_vals = batch_get_polarized_numeric_values(test_symbols, test_alphabet)
    print(f"Polarized numeric values for {test_symbols} in {test_alphabet}: {polarized_vals}")
    """
Fractal Grammar Sequence Generator with Gender-Polarity Integration.

Uses Ray for distributed fractal expansion of symbolic tokens per configurable rules.

Each expanded sequence embeds symbolic polarity via the base numeric utilities
to produce semantically rich linguistic corpora for downstream neural modeling.
"""

import ray
import random
from unified_symbolic_dataset import get_polarized_numeric_value  # Import from first module

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Example fractal grammar expansion rules
# These are illustrative and use Hebrew letters; expand or customize per alphabets.
fractal_rules = {
    'S': [['A', 'B'], ['B', 'A']],  # Start symbol: expands into sequences of A and B
    'A': [['א'], ['ב', 'ג']],       # A expands to either Aleph or Bet+Gimel
    'B': [['ד'], ['ה']],            # B expands to Dalet or He
}

@ray.remote
def fractal_expand(symbol: str, depth: int, rules: dict):
    """
    Recursively expands a symbol using fractal rules to a specified depth.
    At depth 0 or if symbol has no expansion rule, returns symbol as is.
    
    Args:
        symbol (str): The current symbol to expand.
        depth (int): How many levels left to expand.
        rules (dict): Expansion rules mapping symbols to list of expansions (list of lists).
    
    Returns:
        str: The fully expanded symbolic sequence.
    """
    if depth == 0 or symbol not in rules:
        return symbol
    # Randomly select one expansion option
    expansion = random.choice(rules[symbol])
    # Recursively expand all symbols in the selected expansion list
    expanded_children = ray.get([fractal_expand.remote(sym, depth - 1, rules) for sym in expansion])
    return ''.join(expanded_children)

def generate_fractal_corpus(start_symbol: str, depth: int, rules: dict, samples: int):
    """
    Generates multiple fractal expanded sequences from start_symbol for corpus building.
    
    Args:
        start_symbol (str): The root symbol to start expansion.
        depth (int): Recursion depth for fractal expansion.
        rules (dict): The fractal grammar expansion rules.
        samples (int): Number of sequences to generate.
    
    Returns:
        list[str]: List of fractal expanded symbolic sequences.
    """
    futures = [fractal_expand.remote(start_symbol, depth, rules) for _ in range(samples)]
    sequences = ray.get(futures)
    return sequences

# --- Example usage ---

if __name__ == '__main__':
    start = 'S'
    recursion_depth = 4
    sample_count = 5
    
    print(f"Generating {sample_count} fractal sequences from '{start}' with depth {recursion_depth}...")
    fractal_sequences = generate_fractal_corpus(start, recursion_depth, fractal_rules, sample_count)
    
    for i, seq in enumerate(fractal_sequences, 1):
        print(f"Sequence {i}: {seq}")

    # Example: convert sequence symbols to polarized numeric values (Hebrew example)
    alphabet = 'Hebrew'
    for i, seq in enumerate(fractal_sequences, 1):
        polarized_vals = [get_polarized_numeric_value(sym, alphabet) for sym in seq]
        print(f"Polarized values for sequence {i}: {polarized_vals}")"""
PolymathTransformer Neural Language Model Training Module.

Trains a transformer encoder on fractal-generated symbolic sequences with gender-polarity numeric weights.

Includes:
- Vocabulary building and encoding utilities
- Transformer model with positional encoding
- Dataset and DataLoader implementation
- Training and evaluation loops
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

# --- Vocabulary and Encoding Helpers ---

def build_vocab(sequences):
    """
    Builds a token-to-index vocabulary from a list of numeric sequences.
    
    Args:
        sequences (list of list of int): Numeric sequences.
    Returns:
        dict: Mapping from numeric token to index.
    """
    unique_tokens = sorted(set(token for seq in sequences for token in seq))
    return {token: idx for idx, token in enumerate(unique_tokens)}

def encode_sequences(sequences, vocab):
    """
    Encodes sequences of numeric tokens into index sequences for model input.
    
    Args:
        sequences (list of list of int): Numeric sequences.
        vocab (dict): Numeric token to index mapping.
    Returns:
        list of list of int: Indexed sequences.
    """
    return [[vocab[token] for token in seq] for seq in sequences]

# --- Dataset Class for Fractal Sequences ---

class FractalSequenceDataset(Dataset):
    def __init__(self, sequences):
        """
        Args:
            sequences (list of list of int): Indexed numeric sequences.
        """
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns input-output pair for language modeling.
        Inputs: all tokens except last
        Targets: all tokens except first
        """
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

# --- Polymath Transformer Model Definition ---

class PolymathTransformer(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, nhead=8, nlayers=4, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # Positional encoding buffer
        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe.unsqueeze(0))  # Shape (1, max_len, emb_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.fc_out = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len)
        Returns:
            logits (Tensor): Output logits for each token position (batch_size, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :].to(x.device)
        x = self.encoder(x.transpose(0, 1))  # Transformer expects seq_len, batch, emb_dim
        x = x.transpose(0, 1)  # Back to batch, seq_len, emb_dim
        logits = self.fc_out(x)
        return logits

# --- Training Function ---

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

# --- Example Usage ---

if __name__ == '__main__':
    # Sample fractal generated sequences (replace with your real fractal outputs)
    sample_sequences = [
        [1, -2, 3, -4, 5, 6],
        [3, -1, 2, -5, 4, 6],
        [5, 4, -3, 2, -1, 6],
        # Add more sample sequences here
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = build_vocab(sample_sequences)
    encoded_seqs = encode_sequences(sample_sequences, vocab)
    
    dataset = FractalSequenceDataset(encoded_seqs)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = PolymathTransformer(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        """
Sonic Mantra Generation Module.

Generates multi-frequency sonic mantras from fractal linguistic sequences.

Maps symbols to base MIDI notes, converts to frequencies, applies gender-polarity multipliers
to modulate amplitude or phase, and synthesizes sound using pyo.

Requirements:
- pyo library installed (install via: pip install pyo)
"""

from pyo import Server, Sine
import time

# Example base MIDI note mapping for vowels or symbols (customize as needed)
base_midi_map = {
    'a': 60, 'e': 64, 'i': 67, 'o': 69, 'u': 71,
    # Extend with your symbol-to-MIDI mappings
}

# Example gender polarity amplitude multipliers
polarity_amplitude_map = {
    +1: 0.1,   # Masculine/positive amplitude
    -1: 0.05   # Feminine/negative amplitude
}

def midi_to_frequency(midi_note):
    """
    Converts MIDI note number to frequency in Hz.
    """
    return 440.0 * 2 ** ((midi_note - 69) / 12)

class SonicMantraGenerator:
    def __init__(self):
        self.server = Server().boot()
        self.server.start()
        self.oscillators = []

    def play_mantra(self, symbols, polarity_map, duration=3.0):
        """
        Plays a sonic mantra for given symbol list and polarity mapping.

        Args:
            symbols (list[str]): List of symbols (e.g., vowels).
            polarity_map (dict): Map from symbol to polarity (+1/-1).
            duration (float): Duration of sound in seconds.
        """
        # Stop any existing oscillators
        self.stop()

        # Generate frequencies and amplitudes
        for sym in symbols:
            midi_note = base_midi_map.get(sym.lower(), 60)  # Default to middle C
            freq = midi_to_frequency(midi_note)
            polarity = polarity_map.get(sym, +1)
            amp = polarity_amplitude_map.get(polarity, 0.05)
            osc = Sine(freq=freq, mul=amp).out()
            self.oscillators.append(osc)

        time.sleep(duration)
        self.stop()

    def stop(self):
        for osc in self.oscillators:
            osc.stop()
        self.oscillators.clear()
        if self.server.getIsStarted():
            self.server.stop()

# --- Example usage ---

if __name__ == '__main__':
    gen = SonicMantraGenerator()
    sample_mantra = ['a', 'e', 'i', 'o', 'u']
    sample_polarity_map = {sym: (+1 if i % 2 == 0 else -1) for i, sym in enumerate(sample_mantra)}

    print("Playing sample sonic mantra...")
    gen.play_mantra(sample_mantra, sample_polarity_map, duration=5.0)
    """
Sacred Glyph Visualization Module.

Generates fractal glyphs and sacred geometry visualizations based on symbolic sequences.

Uses:
- matplotlib for rendering
- networkx for graph fractal structures and tilings

Requirements:
- matplotlib (`pip install matplotlib`)
- networkx (`pip install networkx`)
"""

import matplotlib.pyplot as plt
import networkx as nx
import math

def create_fractal_tree(depth, pos=(0,0), angle=math.pi/2, length=1.0, angle_delta=math.pi/6):
    """
    Recursively builds a fractal tree graph structure.
    
    Args:
        depth (int): Recursion depth.
        pos (tuple): Position of current node (x,y).
        angle (float): Current angle in radians.
        length (float): Branch length.
        angle_delta (float): Angle deviation between branches.
    
    Returns:
        nx.Graph: Graph with nodes and edges representing fractal.
    """
    G = nx.Graph()
    G.add_node(pos)
    if depth > 0:
        # Left branch
        left_angle = angle + angle_delta
        left_pos = (pos[0] + length * math.cos(left_angle),
                    pos[1] + length * math.sin(left_angle))
        left_subtree = create_fractal_tree(depth - 1, left_pos, left_angle, length * 0.7, angle_delta)
        
        # Right branch
        right_angle = angle - angle_delta
        right_pos = (pos[0] + length * math.cos(right_angle),
                     pos[1] + length * math.sin(right_angle))
        right_subtree = create_fractal_tree(depth - 1, right_pos, right_angle, length * 0.7, angle_delta)
        
        # Merge subtrees
        G = nx.compose(G, left_subtree)
        G = nx.compose(G, right_subtree)
        
        # Add edges to children
        G.add_edge(pos, left_pos)
        G.add_edge(pos, right_pos)
    
    return G

def plot_fractal_glyph(graph, title='Fractal Glyph'):
    """
    Plots the fractal glyph using matplotlib.
    
    Args:
        graph (nx.Graph): Graph structure to plot.
        title (str): Plot title.
    """
    pos = {node: node for node in graph.nodes()}
    plt.figure(figsize=(8,8))
    nx.draw(graph, pos, node_size=30, node_color='black', edge_color='blue')
    plt.title(title)
    plt.axis('off')
    plt.show()

# --- Example usage ---

if __name__ == '__main__':
    depth = 5
    fractal = create_fractal_tree(depth)
    plot_fractal_glyph(fractal, title=f'Sacred Fractal Glyph (Depth={depth})')
    """
API Framework Module using FastAPI.

Exposes:
- Symbolic numeric extraction with polarity multipliers
- Fractal linguistic sequence generation
- Neural model inference (stub implementation)
- Sonic mantra playback trigger
- Sacred glyph visualization generation

Requirements:
- fastapi (`pip install fastapi`)
- uvicorn (`pip install uvicorn`)
- pyo for audio (see earlier requirements)
- matplotlib, networkx for visualization
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Import your previously developed modules (assume properly packaged)
# from symbolic_utils import get_polarized_numeric_value
# from fractal_generator import generate_fractal_corpus
# from polymath_transformer import PolymathTransformer, load_trained_model
# from sonic_mantra import SonicMantraGenerator
# from glyph_visualizer import create_fractal_tree, plot_fractal_glyph # modified for api

app = FastAPI(title="Unified Sacred Linguistics API")

# Data models
class SymbolRequest(BaseModel):
    symbol: str
    alphabet: str

class FractalRequest(BaseModel):
    start_symbol: str
    depth: int
    count: int

class MantraRequest(BaseModel):
    symbols: list[str]
    polarities: dict[str, int]

@app.get("/numeric_value/")
async def numeric_value(symbol: str, alphabet: str):
    # Placeholder example implementation
    val = get_polarized_numeric_value(symbol, alphabet)
    if val == 0:
        raise HTTPException(status_code=404, detail="Symbol or Alphabet not found")
    return {"symbol": symbol, "alphabet": alphabet, "polarized_numeric_value": val}

@app.post("/generate_fractal/")
async def generate_fractal(req: FractalRequest):
    sequences = generate_fractal_corpus(req.start_symbol, req.depth, fractal_rules, req.count)
    return {"count": req.count, "sequences": sequences}

@app.post("/play_mantra/")
async def play_mantra(req: MantraRequest):
    gen = SonicMantraGenerator()
    gen.play_mantra(req.symbols, req.polarities, duration=5.0)
    return {"status": "Mantra playback started"}

@app.get("/visualize_glyph/")
async def visualize_glyph(depth: int = 5):
    graph = create_fractal_tree(depth)
    # Here you would generate an image and return its path or bytes
    # For example, save locally and return filename for download
    filename = f"glyph_depth_{depth}.png"
    plot_fractal_glyph(graph, title=f"Sacred Glyph Depth {depth}")
    return {"filename": filename, "message": f"Glyph visualization generated with depth {depth}"}

@app.get("/")
async def root():
    return {"message": "Unified Sacred Linguistics API is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    """
Chomsky-Style Phrase Structure Grammar and Transformational Syntax Module.

Implements:
- Phrase structure rules (PSRs) for hierarchical sentence construction
- Basic transformational operations (e.g., movement)
- Syntax tree generation compatible with symbolic alphabets (e.g., Hebrew letters)

This acts as a formal syntactic scaffold complementing your fractal semantic framework.
"""

from typing import List, Dict, Union

# Define a sample phrase structure grammar (PSG) for demonstration
# Non-terminals: S (sentence), NP (noun phrase), VP (verb phrase), V (verb), N (noun)
# Terminals: example Hebrew symbols or placeholders
phrase_structure_rules = {
    'S': [['NP', 'VP']],
    'NP': [['D', 'N']],
    'VP': [['V', 'NP']],
    'D': [['ה']],          # Hebrew "the" article
    'N': [['מלך', 'מלכה', 'עבד']],  # 'king', 'queen', 'servant'
    'V': [['רואה', 'אוכל']]          # 'sees', 'eats'
}

class TreeNode:
    def __init__(self, label: str, children: List['TreeNode'] = None):
        self.label = label
        self.children = children or []

    def __repr__(self):
        if not self.children:
            return self.label
        return f"{self.label}({', '.join(map(str, self.children))})"

def expand_psr(symbol: str, rules: Dict[str, List[List[str]]]) -> Union[TreeNode, None]:
    """
    Recursively expands a non-terminal symbol by applying phrase structure rules.
    
    Args:
        symbol (str): The current grammar symbol to expand.
        rules (Dict): Phrase structure production rules.
        
    Returns:
        TreeNode: Root node of syntax tree for this expansion.
    """
    if symbol not in rules:
        # Terminal symbol: return leaf node
        return TreeNode(symbol)
    
    # For simplicity: use first expansion option (can be randomized or iterated)
    expansions = rules[symbol]
    first_expansion = expansions[0]
    children = [expand_psr(sym, rules) for sym in first_expansion]
    return TreeNode(symbol, children)

def apply_transformations(tree: TreeNode) -> TreeNode:
    """
    Applies basic transformations to a syntax tree.
    Example: Move VP to front (Topicalization).
    
    Args:
        tree (TreeNode): The original syntax tree.
        
    Returns:
        TreeNode: Transformed syntax tree.
    """
    if tree.label == 'S' and len(tree.children) == 2:
        # Swap NP and VP to move VP front
        return TreeNode('S', [tree.children[1], tree.children[0]])
    return tree

# --- Example Usage ---

if __name__ == '__main__':
    # Generate base syntax tree from start symbol 'S'
    syntax_tree = expand_psr('S', phrase_structure_rules)
    print("Original Syntax Tree:")
    print(syntax_tree)
    
    # Apply a transformational operation
    transformed_tree = apply_transformations(syntax_tree)
    print("
Transformed Syntax Tree (VP-fronting):")
    print(transformed_tree)
    """
Bridging Parser: Convert Fractal Symbolic Sequences to Syntax Trees using Phrase Structure Rules.
"""

from typing import List, Dict, Optional

class TreeNode:
    def __init__(self, label: str, children: Optional[List['TreeNode']] = None):
        self.label = label
        self.children = children or []

    def __repr__(self):
        if not self.children:
            return self.label
        return f"{self.label}({', '.join(map(str, self.children))})"

# Example mapping: symbol -> phrase category (customize extensively)
symbol_to_category = {
    'א': 'N',  # Hebrew Aleph -> Noun
    'ב': 'D',  # Bet -> Determiner
    'ג': 'V',  # Gimel -> Verb
    # ... add rest of vocab mapping
}

# Phrase Structure Grammar Rules for categories
phrase_structure_rules = {
    'S': [['NP', 'VP']],
    'NP': [['D', 'N']],
    'VP': [['V', 'NP']],
    'D': [['ב']],
    'N': [['א']],
    'V': [['ג']],
}

def parse_sequence_to_tree(seq: List[str], rules: Dict[str, List[List[str]]], category_map: Dict[str, str]) -> TreeNode:
    """
    Parses fractal sequence into syntax tree using categorical and phrase structure expansions.

    Args:
        seq (List[str]): Input fractal symbolic sequence.
        rules (Dict): Phrase structure production rules.
        category_map (Dict): Mapping symbols to phrase categories.

    Returns:
        TreeNode: Root of constructed syntax tree.
    """
    # Helper to expand by phrase rules
    def expand(symbol: str) -> TreeNode:
        if symbol in rules:
            expansion = rules[symbol][0]  # Take first expansion for simplicity
            children = [expand(s) for s in expansion]
            return TreeNode(symbol, children)
        else:
            return TreeNode(symbol)

    # Map sequence symbols to categories for parsing
    seq_categories = [category_map.get(sym, 'N') for sym in seq]

    # For simplicity, create NP and VP manually from fragments in sequence:
    # Here we assume a pattern [D, N, V, D, N], split accordingly.
    # This must be generalized per your grammar complexity.

    if len(seq_categories) < 5:
        return TreeNode('S', [TreeNode('NP', [TreeNode(s) for s in seq_categories])])

    np1 = TreeNode('NP', [TreeNode(seq[0]), TreeNode(seq[1])])
    vp = TreeNode('VP', [TreeNode(seq[2]), TreeNode('NP', [TreeNode(seq[3]), TreeNode(seq[4])])])
    root = TreeNode('S', [np1, vp])
    return root

# --- Usage Example ---

if __name__ == '__main__':
    fractal_seq = ['ב', 'א', 'ג', 'ב', 'א']  # Example symbols mapping to D N V D N
    tree = parse_sequence_to_tree(fractal_seq, phrase_structure_rules, symbol_to_category)
    print("Parsed Syntax Tree:")
    print(tree)
    """
Transformation Engine: Apply complex rewrite rules to syntax trees.
"""

def tree_transform(node: TreeNode) -> TreeNode:
    """
    Applies recursive tree rewriting transformations aligned with fractal archetypes.

    Example transformations:
    - Promote certain NPs to Specifier positions
    - Invert node order in VP for stylistic variation
    - Prune or duplicate subtrees for recursion depth alterations

    Args:
        node (TreeNode): Syntax tree node to transform.

    Returns:
        TreeNode: Transformed tree node.
    """
    # Base case: leaf node
    if not node.children:
        return node

    # Example: invert children order inside a VP
    if node.label == 'VP':
        node.children = [tree_transform(c) for c in reversed(node.children)]
        return node

    # Example: duplicate first NP child (archetypal recursion)
    if node.label == 'S' and node.children:
        children = [tree_transform(c) for c in node.children]
        if children[0].label == 'NP':
            children.insert(1, children[0])  # Duplicate NP
        node.children = children
        return node

    # Recursive transformation for other nodes
    node.children = [tree_transform(c) for c in node.children]
    return node

# --- Usage Example ---

if __name__ == '__main__':
    # Example tree from previous parse
    fractal_seq = ['ב', 'א', 'ג', 'ב', 'א']
    initial_tree = parse_sequence_to_tree(fractal_seq, phrase_structure_rules, symbol_to_category)
    print("Before Transformation:", initial_tree)
    transformed_tree = tree_transform(initial_tree)
    print("After Transformation:", transformed_tree)
    """
Neural Model Dataset Upgrade: Encoding syntax tree structures and polarity annotations jointly.
"""

def flatten_tree(tree: TreeNode) -> List[str]:
    """
    Flattens syntax tree into a sequence with embedded phrase category tokens.
    """
    tokens = []
    def recurse(node):
        tokens.append(f"<{node.label}>")
        for child in node.children:
            recurse(child)
        tokens.append(f"</{node.label}>")
    recurse(tree)
    return tokens

def encode_with_polarity(flat_seq: List[str], polarity_map: Dict[str, int], vocab: Dict[str, int]):
    """
    Encodes flattened tree token list with polarity multipliers from base symbols.

    Phrase category tokens are assigned neutral polarity (1).
    """
    encoded = []
    for token in flat_seq:
        if token.startswith('<') and token.endswith('>'):
            # Phrase category tokens get polarity 1
            encoded.append(vocab.get(token, 0) * 1)
        else:
            pol = polarity_map.get(token, 1)
            encoded.append(vocab.get(token, 0) * pol)
    return encoded

# --- Usage Example ---

if __name__ == '__main__':
    # Flatten and encode example
    fractal_seq = ['ב', 'א', 'ג', 'ב', 'א']
    tree = parse_sequence_to_tree(fractal_seq, phrase_structure_rules, symbol_to_category)
    flat_seq = flatten_tree(tree)
    print("Flattened Tree Sequence:", flat_seq)

    # Example polarity map for symbols only
    polarity_map = {'ב': 1, 'א': -1, 'ג': 1}

    # Build vocab with phrase categories and symbols
    vocab_tokens = list(set(flat_seq + fractal_seq))
    vocab = {tok: idx for idx, tok in enumerate(vocab_tokens)}

    encoded_seq = encode_with_polarity(flat_seq, polarity_map, vocab)
    print("Encoded sequence with polarity:", encoded_seq)