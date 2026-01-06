"""
ToxicityShield - Training Module
Funções de treinamento, avaliação e métricas
"""

import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping:
    """Early stopping para evitar overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.should_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.should_stop


def create_dataloaders(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray,
                       batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Cria DataLoaders para treino, validação e teste"""
    
    train_dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Treina uma época"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.sigmoid(outputs).cpu().detach().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    
    return avg_loss, accuracy


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Avalia modelo"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(probs > 0.5)
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_probs), np.array(all_labels)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 20, lr: float = 0.001, device: torch.device = None,
                early_stopping_patience: int = 5, verbose: bool = True) -> Dict:
    """
    Treina modelo completo
    
    Returns:
        Dict com histórico de treinamento
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Treino
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validação
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Salva histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Scheduler
        scheduler.step(val_loss)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping na época {epoch+1}")
            break
    
    elapsed = time.time() - start_time
    history['training_time'] = elapsed
    print(f"\nTreinamento concluído em {elapsed:.1f}s")
    
    return history


def get_full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    """Calcula todas as métricas de avaliação"""
    y_pred = (y_prob > threshold).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """Imprime métricas formatadas"""
    print(f"\n{'='*50}")
    print(f" {model_name} - Resultados")
    print('='*50)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print('='*50)


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plota curvas de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', color='blue')
    axes[0].plot(history['val_loss'], label='Validation', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', color='blue')
    axes[1].plot(history['val_acc'], label='Validation', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = ['Non-Toxic', 'Toxic'],
                          save_path: Optional[str] = None):
    """Plota matriz de confusão"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "Model",
                   save_path: Optional[str] = None):
    """Plota curva ROC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves_comparison(results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                               save_path: Optional[str] = None):
    """Plota curvas ROC de múltiplos modelos para comparação"""
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, (y_true, y_prob)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2, color=color)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'f1',
                          save_path: Optional[str] = None):
    """Plota comparação de métricas entre modelos"""
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    
    colors = ['#2ecc71' if v == max(values) else '#3498db' for v in values]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Adiciona valores nas barras
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'Model Comparison - {metric.upper()}', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


class ToxicityPredictor:
    """Classe para inferência em produção"""
    
    def __init__(self, model: nn.Module, vocab, preprocessor, max_len: int = 200,
                 device: torch.device = None):
        self.model = model
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """Prediz toxicidade de um texto"""
        # Preprocessa
        cleaned = self.preprocessor.clean_text(text)
        
        # Converte para sequência
        sequence = self.vocab.text_to_sequence(cleaned)
        
        # Padding
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        
        # Predição
        x = torch.LongTensor([sequence]).to(self.device)
        
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        
        return {
            'text': text,
            'cleaned_text': cleaned,
            'is_toxic': prob > 0.5,
            'probability': prob,
            'confidence': abs(prob - 0.5) * 2  # 0-1 scale
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Prediz toxicidade de múltiplos textos"""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    print("Módulo de treinamento carregado com sucesso!")
