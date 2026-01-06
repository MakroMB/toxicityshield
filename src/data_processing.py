"""
ToxicityShield - Data Processing Module
Preprocessamento de texto para detecção de toxicidade
"""

import re
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Preprocessador de texto para limpeza e normalização"""
    
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """Limpa e normaliza texto"""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions e hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove caracteres especiais (mantém letras, números, espaços)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove números
        text = re.sub(r'\d+', '', text)
        
        # Remove espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove stopwords se configurado
        if self.remove_stopwords and self.stop_words:
            words = text.split()
            words = [w for w in words if w not in self.stop_words]
            text = ' '.join(words)
        
        return text
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Processa coluna de texto de um DataFrame"""
        df = df.copy()
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        return df


class Vocabulary:
    """Vocabulário para conversão texto -> sequência de índices"""
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, max_vocab_size: int = 20000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        
    def build_vocab(self, texts: List[str]) -> None:
        """Constrói vocabulário a partir de lista de textos"""
        # Conta frequência de palavras
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Filtra por frequência mínima
        filtered_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]
        
        # Limita tamanho do vocabulário
        filtered_words = filtered_words[:self.max_vocab_size - 2]  # -2 para PAD e UNK
        
        # Cria mappings
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        for idx, word in enumerate(filtered_words, start=2):
            self.word2idx[word] = idx
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"Vocabulário construído: {len(self.word2idx)} palavras")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Converte texto para sequência de índices"""
        words = text.split()
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) for word in words]
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Converte sequência de índices para texto"""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in sequence]
        return ' '.join(words)
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: str) -> None:
        """Salva vocabulário em arquivo"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Carrega vocabulário de arquivo"""
        vocab = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            vocab.word2idx = data['word2idx']
            vocab.idx2word = data['idx2word']
            vocab.word_freq = data['word_freq']
        return vocab


def pad_sequences(sequences: List[List[int]], max_len: int, 
                  padding: str = 'post', truncating: str = 'post') -> np.ndarray:
    """
    Aplica padding/truncamento em sequências
    
    Args:
        sequences: Lista de sequências de índices
        max_len: Comprimento máximo
        padding: 'pre' ou 'post' - onde adicionar padding
        truncating: 'pre' ou 'post' - onde truncar
    
    Returns:
        Array numpy com sequências padded
    """
    result = np.zeros((len(sequences), max_len), dtype=np.int64)
    
    for i, seq in enumerate(sequences):
        if len(seq) == 0:
            continue
            
        # Truncar se necessário
        if len(seq) > max_len:
            if truncating == 'pre':
                seq = seq[-max_len:]
            else:
                seq = seq[:max_len]
        
        # Aplicar padding
        if padding == 'pre':
            result[i, -len(seq):] = seq
        else:
            result[i, :len(seq)] = seq
    
    return result


def load_kaggle_data(filepath: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Carrega dados do Kaggle Toxic Comment Challenge
    
    Args:
        filepath: Caminho para o CSV
        sample_size: Se definido, retorna amostra aleatória
    
    Returns:
        DataFrame com colunas: text, toxic, labels
    """
    df = pd.read_csv(filepath)
    
    # Colunas de toxicidade do dataset Jigsaw
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # Verifica se as colunas existem
    available_toxic_cols = [col for col in toxic_columns if col in df.columns]
    
    if not available_toxic_cols:
        raise ValueError(f"Colunas de toxicidade não encontradas. Colunas disponíveis: {df.columns.tolist()}")
    
    # Identifica coluna de texto
    text_column = 'comment_text' if 'comment_text' in df.columns else df.columns[1]
    
    # Cria label binária (1 se qualquer tipo de toxicidade)
    df['toxic_binary'] = (df[available_toxic_cols].sum(axis=1) > 0).astype(int)
    
    # Renomeia para padronizar
    df = df.rename(columns={text_column: 'text'})
    
    # Amostragem se solicitada
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Dataset carregado: {len(df)} amostras")
    print(f"Distribuição: {df['toxic_binary'].value_counts().to_dict()}")
    
    return df


def prepare_data(df: pd.DataFrame, vocab: Vocabulary, max_len: int = 200,
                test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
    """
    Prepara dados para treinamento
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    # Converte textos para sequências
    sequences = [vocab.text_to_sequence(text) for text in df['cleaned_text']]
    
    # Aplica padding
    X = pad_sequences(sequences, max_len)
    y = df['toxic_binary'].values
    
    # Split train/temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42, stratify=y
    )
    
    # Split val/test
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42, stratify=y_temp
    )
    
    print(f"Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Teste do módulo
    print("Testando TextPreprocessor...")
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "You're such an IDIOT!!! @username http://example.com",
        "Great game! Well played everyone 👏",
        "omg this is so #toxic 123"
    ]
    
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        print(f"Original: {text}")
        print(f"Limpo:    {cleaned}\n")
