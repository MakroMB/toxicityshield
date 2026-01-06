"""
ToxicityShield - Gradio Interface
Interface web para detecção de toxicidade em tempo real
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import re
import pickle
from pathlib import Path


# ============================================================================
# MODELOS (cópia simplificada para standalone)
# ============================================================================

class LSTMClassifier(nn.Module):
    """Classificador LSTM bidirecional com Attention"""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context).squeeze(-1)


# ============================================================================
# PREPROCESSAMENTO
# ============================================================================

class TextPreprocessor:
    """Preprocessador de texto"""
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


class Vocabulary:
    """Vocabulário para conversão texto -> índices"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
    
    def text_to_sequence(self, text):
        words = text.split()
        return [self.word2idx.get(word, 1) for word in words]  # 1 = UNK
    
    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
            vocab.word2idx = data['word2idx']
            vocab.idx2word = data['idx2word']
        return vocab


# ============================================================================
# PREDICTOR
# ============================================================================

class ToxicityPredictor:
    """Preditor de toxicidade"""
    
    def __init__(self, model, vocab, preprocessor, max_len=150):
        self.model = model
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.model.eval()
    
    def predict(self, text):
        if not text or not text.strip():
            return {"error": "Texto vazio"}
        
        # Preprocessa
        cleaned = self.preprocessor.clean_text(text)
        if not cleaned:
            return {"error": "Texto inválido após limpeza"}
        
        # Converte para sequência
        sequence = self.vocab.text_to_sequence(cleaned)
        
        # Padding
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        
        # Predição
        x = torch.LongTensor([sequence])
        
        with torch.no_grad():
            logit = self.model(x)
            prob = torch.sigmoid(logit).item()
        
        return {
            'probability': prob,
            'is_toxic': prob > 0.5,
            'confidence': abs(prob - 0.5) * 2,
            'cleaned_text': cleaned
        }


# ============================================================================
# DEMO MODE (sem modelo treinado)
# ============================================================================

class DemoPredictor:
    """Preditor demo usando heurísticas simples (para demonstração)"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
        # Palavras tóxicas comuns para demo
        self.toxic_words = {
            'idiot', 'stupid', 'dumb', 'trash', 'noob', 'loser', 'suck',
            'hate', 'kill', 'die', 'worst', 'terrible', 'awful', 'pathetic',
            'moron', 'retard', 'fool', 'jerk', 'shut', 'stfu', 'gtfo',
            'cancer', 'toxic', 'garbage', 'useless', 'worthless', 'delete',
            'uninstall', 'bad', 'horrible', 'disgusting', 'ugly', 'fat',
            'kys', 'neck', 'rope', 'brain', 'dead', 'braindead'
        }
        
        self.positive_words = {
            'good', 'great', 'nice', 'awesome', 'amazing', 'well', 'wp',
            'gg', 'gj', 'thanks', 'thank', 'please', 'help', 'love',
            'beautiful', 'excellent', 'fantastic', 'wonderful', 'best',
            'fun', 'enjoy', 'happy', 'luck', 'gl', 'hf', 'friend'
        }
    
    def predict(self, text):
        if not text or not text.strip():
            return {"error": "Texto vazio"}
        
        cleaned = self.preprocessor.clean_text(text)
        words = set(cleaned.split())
        
        # Conta matches
        toxic_count = len(words & self.toxic_words)
        positive_count = len(words & self.positive_words)
        
        # Calcula score
        if toxic_count + positive_count == 0:
            prob = 0.3  # Neutro
        else:
            prob = toxic_count / (toxic_count + positive_count + 1)
            prob = min(0.95, max(0.05, prob * 1.5))  # Ajusta range
        
        # Boost para frases muito agressivas
        aggressive_patterns = ['kill yourself', 'kys', 'go die', 'delete yourself']
        for pattern in aggressive_patterns:
            if pattern in cleaned:
                prob = max(prob, 0.95)
        
        return {
            'probability': prob,
            'is_toxic': prob > 0.5,
            'confidence': abs(prob - 0.5) * 2,
            'cleaned_text': cleaned
        }


# ============================================================================
# INTERFACE GRADIO
# ============================================================================

def create_interface():
    """Cria interface Gradio"""
    
    # Tenta carregar modelo treinado, senão usa demo
    model_path = Path("models/toxicity_lstm_final.pt")
    vocab_path = Path("models/vocab.pkl")
    
    if model_path.exists() and vocab_path.exists():
        print("✅ Carregando modelo treinado...")
        checkpoint = torch.load(model_path, map_location='cpu')
        model = LSTMClassifier(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint.get('embedding_dim', 128),
            hidden_dim=checkpoint.get('hidden_dim', 128)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        vocab = Vocabulary.load(vocab_path)
        predictor = ToxicityPredictor(model, vocab, TextPreprocessor())
        mode = "🤖 Modelo LSTM Treinado"
    else:
        print("⚠️ Modelo não encontrado, usando modo demo...")
        predictor = DemoPredictor()
        mode = "🎮 Modo Demo (Heurístico)"
    
    def analyze_text(text):
        """Analisa toxicidade do texto"""
        result = predictor.predict(text)
        
        if "error" in result:
            return (
                "⚠️ Erro",
                result["error"],
                "",
                0
            )
        
        prob = result['probability']
        is_toxic = result['is_toxic']
        confidence = result['confidence']
        
        # Determina status e cor
        if prob < 0.3:
            status = "🟢 SEGURO"
            emoji = "✅"
            color = "#2ecc71"
        elif prob < 0.5:
            status = "🟡 CUIDADO"
            emoji = "⚠️"
            color = "#f39c12"
        elif prob < 0.7:
            status = "🟠 TÓXICO"
            emoji = "🚫"
            color = "#e67e22"
        else:
            status = "🔴 MUITO TÓXICO"
            emoji = "☠️"
            color = "#e74c3c"
        
        # Formata detalhes
        details = f"""
**Probabilidade de Toxicidade:** {prob:.1%}
**Confiança:** {confidence:.1%}
**Texto Limpo:** _{result['cleaned_text']}_
        """
        
        return (
            status,
            f"{emoji} {'Esta mensagem foi classificada como TÓXICA' if is_toxic else 'Esta mensagem parece SEGURA'}",
            details,
            prob
        )
    
    def analyze_batch(texts):
        """Analisa múltiplas mensagens"""
        if not texts.strip():
            return "Digite algumas mensagens (uma por linha)"
        
        lines = [l.strip() for l in texts.split('\n') if l.strip()]
        
        results = []
        for line in lines[:10]:  # Limita a 10
            result = predictor.predict(line)
            if "error" not in result:
                prob = result['probability']
                if prob < 0.3:
                    icon = "🟢"
                elif prob < 0.5:
                    icon = "🟡"
                elif prob < 0.7:
                    icon = "🟠"
                else:
                    icon = "🔴"
                results.append(f"{icon} **{prob:.0%}** — {line[:50]}...")
        
        return "\n".join(results) if results else "Nenhum resultado"
    
    # CSS customizado
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    """
    
    # Interface
    with gr.Blocks(css=css, title="ToxicityShield") as demo:
        gr.Markdown(f"""
        # 🛡️ ToxicityShield
        ### Detecção de Toxicidade em Chats de Jogos
        
        **Modo atual:** {mode}
        
        ---
        """)
        
        with gr.Tab("🔍 Análise Individual"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Digite uma mensagem de chat",
                        placeholder="Ex: gg wp, great game everyone!",
                        lines=3
                    )
                    analyze_btn = gr.Button("🔍 Analisar", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    status_output = gr.Textbox(label="Status", interactive=False)
                    result_output = gr.Textbox(label="Resultado", interactive=False)
                    prob_slider = gr.Slider(
                        minimum=0, maximum=1, value=0,
                        label="Nível de Toxicidade",
                        interactive=False
                    )
            
            details_output = gr.Markdown(label="Detalhes")
            
            analyze_btn.click(
                analyze_text,
                inputs=[input_text],
                outputs=[status_output, result_output, details_output, prob_slider]
            )
            
            gr.Markdown("### 📝 Exemplos")
            gr.Examples(
                examples=[
                    ["gg wp, great game everyone!"],
                    ["nice shot! that was impressive"],
                    ["you're so bad at this game, uninstall noob"],
                    ["can someone help me with the objective?"],
                    ["ur trash delete the game idiot"],
                    ["good luck have fun!"],
                    ["go kill yourself stupid"],
                    ["well played, close match!"]
                ],
                inputs=input_text
            )
        
        with gr.Tab("📊 Análise em Lote"):
            gr.Markdown("Analise várias mensagens de uma vez (uma por linha)")
            
            batch_input = gr.Textbox(
                label="Mensagens (uma por linha)",
                placeholder="gg wp\\nyou suck\\nnice game\\ndelete noob",
                lines=8
            )
            batch_btn = gr.Button("📊 Analisar Todas", variant="primary")
            batch_output = gr.Markdown(label="Resultados")
            
            batch_btn.click(
                analyze_batch,
                inputs=[batch_input],
                outputs=[batch_output]
            )
        
        with gr.Tab("ℹ️ Sobre"):
            gr.Markdown("""
            ## Sobre o ToxicityShield
            
            Este projeto utiliza **Deep Learning** para detectar toxicidade em mensagens de chat,
            especialmente focado em contextos de jogos online.
            
            ### 🧠 Modelos Disponíveis
            
            | Modelo | Descrição |
            |--------|-----------|
            | LSTM | Rede recorrente bidirecional com attention |
            | GRU | Similar ao LSTM, menos parâmetros |
            | TextCNN | CNN 1D com múltiplos kernels |
            | Mini-Transformer | Encoder com self-attention |
            
            ### 📊 Métricas (quando treinado)
            
            - **Accuracy:** ~95%
            - **F1-Score:** ~0.89
            - **ROC-AUC:** ~0.98
            
            ### 🎮 Casos de Uso
            
            - Moderação automática de chats em jogos
            - Filtragem de comentários em redes sociais
            - Detecção de cyberbullying
            - Análise de sentimento em comunidades
            
            ---
            
            **Desenvolvido por Markko** | Projeto para Portfolio de ML
            """)
        
        gr.Markdown("""
        ---
        <center>
        
        🛡️ **ToxicityShield** — Deep Learning for Toxicity Detection
        
        </center>
        """)
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Mude para True para criar link público
        show_error=True
    )
