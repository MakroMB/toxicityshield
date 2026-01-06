"""
ToxicityShield - Models Module
Arquiteturas de Deep Learning para classificação de toxicidade
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LSTMClassifier(nn.Module):
    """
    Classificador baseado em LSTM bidirecional
    
    Arquitetura:
    Embedding -> LSTM Bidirecional -> Attention -> FC -> Sigmoid
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128, 
                 hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention
        self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x: (batch, seq_len)
        
        # Embedding: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(embedded)
        
        # Attention weights: (batch, seq_len, 1)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted sum: (batch, hidden_dim * num_directions)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Output: (batch, 1)
        out = self.fc(context)
        
        return out.squeeze(-1)


class GRUClassifier(nn.Module):
    """
    Classificador baseado em GRU bidirecional
    Similar ao LSTM mas com menos parâmetros
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # GRU output
        gru_out, hidden = self.gru(embedded)
        
        # Usa último hidden state de ambas direções
        if self.num_directions == 2:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        out = self.fc(hidden)
        return out.squeeze(-1)


class TextCNN(nn.Module):
    """
    CNN para classificação de texto (Kim 2014)
    
    Arquitetura:
    Embedding -> Conv1D (múltiplos kernels) -> MaxPool -> FC -> Sigmoid
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_filters: int = 100, filter_sizes: tuple = (2, 3, 4, 5),
                 dropout: float = 0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Múltiplas convoluções com diferentes tamanhos de kernel
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters * len(filter_sizes), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len)
        
        # Embedding: (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Transpose para Conv1d: (batch, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Convolução + ReLU + MaxPool para cada tamanho de filtro
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch, num_filters, seq_len - fs + 1)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))
        
        # Concatena saídas: (batch, num_filters * len(filter_sizes))
        concat = torch.cat(conv_outputs, dim=1)
        
        # FC
        out = self.fc(concat)
        return out.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Positional encoding para Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Cria positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MiniTransformer(nn.Module):
    """
    Mini-Transformer para classificação de texto
    
    Arquitetura simplificada com:
    - Embedding + Positional Encoding
    - 2 Transformer Encoder layers
    - Global average pooling + FC
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 2,
                 ff_dim: int = 256, max_len: int = 200, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len, dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classificador
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len)
        
        # Cria máscara de padding
        padding_mask = (x == 0)  # True onde há padding
        
        # Embedding + Positional encoding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = self.pos_encoding(embedded)
        
        # Transformer
        transformed = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # Global average pooling (ignora padding)
        mask = ~padding_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        masked_output = transformed * mask.float()
        pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # FC
        out = self.fc(pooled)
        return out.squeeze(-1)


def get_model(model_name: str, vocab_size: int, **kwargs) -> nn.Module:
    """
    Factory function para criar modelos
    
    Args:
        model_name: 'lstm', 'gru', 'cnn', ou 'transformer'
        vocab_size: Tamanho do vocabulário
        **kwargs: Argumentos específicos do modelo
    
    Returns:
        Modelo instanciado
    """
    models = {
        'lstm': LSTMClassifier,
        'gru': GRUClassifier,
        'cnn': TextCNN,
        'transformer': MiniTransformer
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Modelo '{model_name}' não encontrado. Opções: {list(models.keys())}")
    
    return models[model_name.lower()](vocab_size, **kwargs)


def count_parameters(model: nn.Module) -> int:
    """Conta parâmetros treináveis do modelo"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Teste dos modelos
    print("Testando modelos...")
    
    vocab_size = 10000
    batch_size = 32
    seq_len = 100
    
    # Input de teste
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    models_to_test = ['lstm', 'gru', 'cnn', 'transformer']
    
    for name in models_to_test:
        model = get_model(name, vocab_size)
        output = model(x)
        params = count_parameters(model)
        print(f"{name.upper():12} | Output shape: {output.shape} | Params: {params:,}")
