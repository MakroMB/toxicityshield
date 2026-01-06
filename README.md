# 🛡️ ToxicityShield

**Deep Learning for Toxicity Detection in Game Chats**

Sistema de detecção de toxicidade em tempo real usando NLP e Deep Learning para identificar e filtrar mensagens tóxicas em chats de jogos online.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Visão Geral

ToxicityShield é um projeto de Machine Learning que utiliza técnicas avançadas de NLP e Deep Learning para detectar automaticamente toxicidade em mensagens de chat.

### Casos de Uso
- 🎮 Moderação automática de chats em jogos online
- 💬 Filtragem de comentários em redes sociais
- 🔍 Detecção de cyberbullying
- 📊 Análise de sentimento em comunidades online

---

## ✨ Features

- ✅ **Multiple Model Architectures**: LSTM, GRU, CNN, e Transformer
- ✅ **Baseline Comparisons**: Logistic Regression e SVM com TF-IDF
- ✅ **Complete Pipeline**: Pré-processamento, treinamento, avaliação
- ✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- ✅ **Visualization Tools**: Confusion matrices, ROC curves, training history
- ✅ **Production-Ready**: Modelos salvos prontos para deployment

---

## 🏗️ Arquitetura

```
Raw Text → Preprocessing → Tokenization → Model → Classification
    ↓            ↓              ↓           ↓           ↓
 Cleaning   Stopwords    Vocabulary    LSTM/CNN    Toxic/Safe
            Removal      Encoding    Transformer
```

### Modelos Disponíveis

| Modelo | Tipo | Descrição |
|--------|------|-----------|
| **LSTM** | Deep Learning | Bidirecional com Attention |
| **GRU** | Deep Learning | Bidirecional, menos parâmetros |
| **TextCNN** | Deep Learning | CNN 1D com múltiplos kernels |
| **Mini-Transformer** | Deep Learning | Encoder com self-attention |
| Logistic Regression | Baseline | TF-IDF + classificador linear |
| SVM | Baseline | TF-IDF + Support Vector Machine |

---

## 📂 Estrutura do Projeto

```
toxicityshield/
├── data/
│   ├── raw/                    # CSV do Kaggle
│   └── processed/              # Dados processados
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Pipeline de preprocessamento
│   ├── models.py               # Arquiteturas dos modelos
│   └── training.py             # Training & evaluation
├── notebooks/
│   └── complete_project.ipynb  # Notebook completo
├── models/                     # Modelos salvos
├── results/                    # Gráficos e métricas
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone e Instale

```bash
git clone https://github.com/seu-usuario/toxicityshield.git
cd toxicityshield
pip install -r requirements.txt
```

### 2. Baixe o Dataset

1. Acesse: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
2. Baixe `train.csv`
3. Coloque em `data/raw/train.csv`

### 3. Execute o Notebook

```bash
jupyter notebook notebooks/complete_project.ipynb
```

---

## 📊 Dataset

**Jigsaw Toxic Comment Classification Challenge**
- ~160K comentários rotulados
- 6 categorias: toxic, severe_toxic, obscene, threat, insult, identity_hate
- Convertido para classificação binária (toxic vs non-toxic)

---

## 💻 Uso

### Inferência Rápida

```python
from src import ToxicityPredictor, TextPreprocessor, Vocabulary, get_model
import torch

# Carrega componentes
vocab = Vocabulary.load('models/vocab.pkl')
preprocessor = TextPreprocessor()

# Carrega modelo
checkpoint = torch.load('models/toxicity_lstm_final.pt')
model = get_model('lstm', checkpoint['vocab_size'])
model.load_state_dict(checkpoint['model_state_dict'])

# Cria preditor
predictor = ToxicityPredictor(model, vocab, preprocessor)

# Prediz
result = predictor.predict("you're trash at this game noob")
print(result)
# {'is_toxic': True, 'probability': 0.92, 'confidence': 0.84}
```

---

## 🎓 Para o Currículo

```
ToxicityShield — Deep Learning for Toxicity Detection in Game Chats
Python, PyTorch, NLP, scikit-learn

• Trained deep learning models (LSTM, GRU, CNN, Transformer) on 50K+ chat
  messages to classify toxic vs. non-toxic content
  
• Compared recurrent networks (LSTM/GRU) and transformer-based architectures
  against classical baselines (Logistic Regression, SVM)
  
• Implemented preprocessing pipeline (tokenization, embeddings, padding) and
  evaluation with precision/recall metrics and ROC curves
  
• Built production-ready inference system for real-time toxicity detection
```

---

## 📝 Próximos Passos

- [ ] Fine-tuning com BERT/RoBERTa
- [ ] Multi-label classification (tipos de toxicidade)
- [ ] Embeddings pré-treinados (GloVe, FastText)
- [ ] API REST com Flask/FastAPI
- [ ] Deploy com Docker + AWS Lambda
- [ ] Interface demo com Gradio/Streamlit

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 🖥️ Interface Web

O projeto inclui duas opções de interface:

### Gradio (recomendado)
```bash
python app.py
# Acesse: http://localhost:7860
```

### Streamlit
```bash
streamlit run app_streamlit.py
# Acesse: http://localhost:8501
```

Ambas funcionam em **modo demo** (heurístico) se o modelo não estiver treinado,
ou carregam o **modelo LSTM** automaticamente se disponível em `models/`.

---

**Desenvolvido por Markko** | [GitHub](https://github.com/seu-usuario)
