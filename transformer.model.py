import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Archetypal category mappings
POLARITY_TO_IDX = {"Yang": 0, "Yin": 1, "Balanced": 2, "None": 3}
ELEMENT_TO_IDX = {"Air": 0, "Water": 1, "Fire": 2, "Earth": 3, "Ether": 4, "None": 5}
GENDER_TO_IDX = {"male": 0, "female": 1, "neutral": 2, "unknown": 3}

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class ArchetypalTransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.polarity_embedding = nn.Embedding(len(POLARITY_TO_IDX), embedding_dim)
        self.element_embedding = nn.Embedding(len(ELEMENT_TO_IDX), embedding_dim)
        self.gender_embedding = nn.Embedding(len(GENDER_TO_IDX), embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)

    def forward(self, token_ids, polarity_ids, element_ids, gender_ids):
        token_emb = self.token_embedding(token_ids)
        polarity_emb = self.polarity_embedding(polarity_ids)
        element_emb = self.element_embedding(element_ids)
        gender_emb = self.gender_embedding(gender_ids)
        combined = token_emb + polarity_emb + element_emb + gender_emb
        combined = self.positional_encoding(combined)
        return combined

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.embedding = ArchetypalTransformerEmbedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, polarity_ids, element_ids, gender_ids):
        x = self.embedding(token_ids, polarity_ids, element_ids, gender_ids)  # (batch, seq_len, embedding_dim)
        x = x.transpose(0, 1)  # (seq_len, batch, embedding_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Back to (batch, seq_len, embedding_dim)
        output = self.output_layer(x)  # (batch, seq_len, vocab_size)
        return output

def prepare_transformer_inputs(fractal_tokens, token_to_id):
    import torch
    seq_len = len(fractal_tokens)
    token_ids = torch.zeros(1, seq_len, dtype=torch.long)
    polarity_ids = torch.zeros(1, seq_len, dtype=torch.long)
    element_ids = torch.zeros(1, seq_len, dtype=torch.long)
    gender_ids = torch.zeros(1, seq_len, dtype=torch.long)

    for i, token in enumerate(fractal_tokens):
        token_ids[0, i] = token_to_id.get(token['symbol'], token_to_id.get('<unk>', 0))
        polarity_ids[0, i] = POLARITY_TO_IDX.get(token.get('polarity', 'None'), POLARITY_TO_IDX['None'])
        element_ids[0, i] = ELEMENT_TO_IDX.get(token.get('element', 'None'), ELEMENT_TO_IDX['None'])
        gender_ids[0, i] = GENDER_TO_IDX.get(token.get('gender', 'unknown'), GENDER_TO_IDX['unknown'])

    return token_ids, polarity_ids, element_ids, gender_ids

class RitualSymbolicDataset(Dataset):
    def __init__(self, sequences, token_to_id):
        self.sequences = sequences
        self.token_to_id = token_to_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        fractal_tokens = self.sequences[idx]
        token_ids, polarity_ids, element_ids, gender_ids = prepare_transformer_inputs(fractal_tokens, self.token_to_id)
        return {
            'token_ids': token_ids.squeeze(0),
            'polarity_ids': polarity_ids.squeeze(0),
            'element_ids': element_ids.squeeze(0),
            'gender_ids': gender_ids.squeeze(0),
        }

def train_transformer(model, dataset, epochs=10, batch_size=8, lr=1e-4):
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is pad token idx if used
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            token_ids = batch['token_ids'].to(device)
            polarity_ids = batch['polarity_ids'].to(device)
            element_ids = batch['element_ids'].to(device)
            gender_ids = batch['gender_ids'].to(device)

            output = model(token_ids, polarity_ids, element_ids, gender_ids)
            target = token_ids[:, 1:]
            output = output[:, :-1, :]
            loss = criterion(output.reshape(-1, output.shape[-1]), target.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}")