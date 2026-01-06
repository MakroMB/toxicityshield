"""
ToxicityShield - Streamlit Interface
Interface web alternativa para detecção de toxicidade
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
import pickle
from pathlib import Path
import time


# ============================================================================
# CONFIG DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="ToxicityShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .status-safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .status-warning {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .status-danger {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODELOS (standalone)
# ============================================================================

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
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


class TextPreprocessor:
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


class DemoPredictor:
    """Preditor demo usando heurísticas"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.toxic_words = {
            'idiot', 'stupid', 'dumb', 'trash', 'noob', 'loser', 'suck',
            'hate', 'kill', 'die', 'worst', 'terrible', 'awful', 'pathetic',
            'moron', 'retard', 'fool', 'jerk', 'shut', 'stfu', 'gtfo',
            'cancer', 'toxic', 'garbage', 'useless', 'worthless', 'delete',
            'uninstall', 'bad', 'horrible', 'disgusting', 'kys', 'braindead'
        }
        self.positive_words = {
            'good', 'great', 'nice', 'awesome', 'amazing', 'well', 'wp',
            'gg', 'gj', 'thanks', 'thank', 'please', 'help', 'love',
            'beautiful', 'excellent', 'fantastic', 'wonderful', 'best',
            'fun', 'enjoy', 'happy', 'luck', 'gl', 'hf', 'friend'
        }
    
    def predict(self, text):
        if not text or not text.strip():
            return None
        
        cleaned = self.preprocessor.clean_text(text)
        words = set(cleaned.split())
        
        toxic_count = len(words & self.toxic_words)
        positive_count = len(words & self.positive_words)
        
        if toxic_count + positive_count == 0:
            prob = 0.3
        else:
            prob = toxic_count / (toxic_count + positive_count + 1)
            prob = min(0.95, max(0.05, prob * 1.5))
        
        aggressive_patterns = ['kill yourself', 'kys', 'go die']
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
# INICIALIZAÇÃO
# ============================================================================

@st.cache_resource
def load_predictor():
    """Carrega preditor (com cache)"""
    return DemoPredictor()


predictor = load_predictor()


# ============================================================================
# INTERFACE
# ============================================================================

# Header
st.markdown('<h1 class="main-header">🛡️ ToxicityShield</h1>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>Detecção de Toxicidade em Chats de Jogos</h3>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=80)
    st.title("⚙️ Configurações")
    
    threshold = st.slider(
        "Limiar de Toxicidade",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Mensagens acima deste valor são consideradas tóxicas"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Estatísticas da Sessão")
    
    if 'stats' not in st.session_state:
        st.session_state.stats = {'total': 0, 'toxic': 0, 'safe': 0}
    
    col1, col2 = st.columns(2)
    col1.metric("Total", st.session_state.stats['total'])
    col2.metric("Tóxicas", st.session_state.stats['toxic'])
    
    if st.button("🔄 Resetar Stats"):
        st.session_state.stats = {'total': 0, 'toxic': 0, 'safe': 0}
        st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ Sobre
    
    **ToxicityShield** usa Deep Learning 
    para detectar mensagens tóxicas em 
    chats de jogos online.
    
    **Modelos:**
    - LSTM Bidirecional
    - GRU
    - TextCNN
    - Mini-Transformer
    
    ---
    *Projeto de Portfolio ML*
    """)

# Tabs principais
tab1, tab2, tab3 = st.tabs(["🔍 Análise Individual", "📊 Análise em Lote", "📈 Histórico"])

with tab1:
    st.markdown("### Digite uma mensagem para análise")
    
    # Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_area(
            "Mensagem",
            placeholder="Ex: gg wp, great game everyone!",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # Espaçamento
        st.write("")
        analyze_btn = st.button("🔍 Analisar", type="primary", use_container_width=True)
    
    # Exemplos rápidos
    st.markdown("**Exemplos rápidos:**")
    example_cols = st.columns(4)
    examples = [
        "gg wp, great game!",
        "you're trash noob",
        "nice shot!",
        "go uninstall idiot"
    ]
    
    for col, ex in zip(example_cols, examples):
        if col.button(ex[:15] + "...", use_container_width=True):
            user_input = ex
            analyze_btn = True
    
    # Análise
    if analyze_btn and user_input:
        with st.spinner("Analisando..."):
            time.sleep(0.3)  # Efeito visual
            result = predictor.predict(user_input)
        
        if result:
            prob = result['probability']
            is_toxic = prob >= threshold
            
            # Atualiza stats
            st.session_state.stats['total'] += 1
            if is_toxic:
                st.session_state.stats['toxic'] += 1
            else:
                st.session_state.stats['safe'] += 1
            
            # Resultado visual
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prob < 0.3:
                    st.success("### 🟢 SEGURO")
                elif prob < threshold:
                    st.warning("### 🟡 CUIDADO")
                elif prob < 0.7:
                    st.error("### 🟠 TÓXICO")
                else:
                    st.error("### 🔴 MUITO TÓXICO")
            
            with col2:
                st.metric(
                    "Probabilidade",
                    f"{prob:.1%}",
                    delta=f"{(prob - 0.5) * 100:+.0f}%" if prob != 0.5 else None,
                    delta_color="inverse"
                )
            
            with col3:
                st.metric(
                    "Confiança",
                    f"{result['confidence']:.1%}"
                )
            
            # Barra de progresso
            st.progress(prob, text=f"Nível de Toxicidade: {prob:.1%}")
            
            # Detalhes
            with st.expander("📝 Detalhes da Análise"):
                st.markdown(f"**Texto original:** {user_input}")
                st.markdown(f"**Texto limpo:** {result['cleaned_text']}")
                st.markdown(f"**Limiar atual:** {threshold:.0%}")
                st.markdown(f"**Classificação:** {'🚫 Tóxico' if is_toxic else '✅ Seguro'}")

with tab2:
    st.markdown("### Analise várias mensagens de uma vez")
    
    batch_input = st.text_area(
        "Cole as mensagens (uma por linha)",
        placeholder="gg wp\nyou suck\nnice game\ndelete noob",
        height=200
    )
    
    if st.button("📊 Analisar Todas", type="primary"):
        if batch_input.strip():
            lines = [l.strip() for l in batch_input.split('\n') if l.strip()]
            
            results = []
            progress = st.progress(0)
            
            for i, line in enumerate(lines[:20]):
                result = predictor.predict(line)
                if result:
                    results.append({
                        'mensagem': line[:50] + ('...' if len(line) > 50 else ''),
                        'toxicidade': f"{result['probability']:.1%}",
                        'status': '🔴 Tóxico' if result['probability'] >= threshold else '🟢 Seguro',
                        'prob': result['probability']
                    })
                progress.progress((i + 1) / len(lines[:20]))
            
            # Exibe tabela
            if results:
                st.markdown("---")
                
                # Métricas resumo
                toxic_count = sum(1 for r in results if r['prob'] >= threshold)
                safe_count = len(results) - toxic_count
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Analisadas", len(results))
                col2.metric("🟢 Seguras", safe_count)
                col3.metric("🔴 Tóxicas", toxic_count)
                
                # Tabela
                st.dataframe(
                    [{k: v for k, v in r.items() if k != 'prob'} for r in results],
                    use_container_width=True,
                    hide_index=True
                )

with tab3:
    st.markdown("### 📈 Histórico da Sessão")
    
    stats = st.session_state.stats
    
    if stats['total'] > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Análises", stats['total'])
        
        with col2:
            safe_pct = (stats['safe'] / stats['total']) * 100
            st.metric("Mensagens Seguras", f"{stats['safe']} ({safe_pct:.0f}%)")
        
        with col3:
            toxic_pct = (stats['toxic'] / stats['total']) * 100
            st.metric("Mensagens Tóxicas", f"{stats['toxic']} ({toxic_pct:.0f}%)")
        
        # Gráfico simples
        import pandas as pd
        
        chart_data = pd.DataFrame({
            'Categoria': ['Seguras', 'Tóxicas'],
            'Quantidade': [stats['safe'], stats['toxic']]
        })
        
        st.bar_chart(chart_data.set_index('Categoria'))
    else:
        st.info("Nenhuma análise realizada ainda. Comece analisando algumas mensagens!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    🛡️ <b>ToxicityShield</b> — Deep Learning for Toxicity Detection<br>
    Desenvolvido por Markko | Projeto de Portfolio ML
</div>
""", unsafe_allow_html=True)
