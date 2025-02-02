import streamlit as st
import time
import random
import numpy as np
import pandas as pd
import plotly.express as px
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization
from sklearn.decomposition import PCA
import re

# =============================================================================
# Custom CSS Styling
# =============================================================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .header {
        background-color: #2c3e50;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .explanation-box {
        background-color: #e9f5ff;
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        margin: 1rem 0;
    }
    .debug-console {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .footer {
        position: relative !important;
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        margin-top: 3rem;
        border-radius: 5px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions and Classes 
# =============================================================================
tokenizer = TreebankWordTokenizer()

def load_embeddings(vocab):
    """Generate random embeddings for demonstration purposes."""
    return {word: np.random.rand(50) for word in vocab}

class SimpleLanguageModel:
    """Trigram-based language model with training and generation capabilities."""
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, text):
        tokens = tokenizer.tokenize(text.lower())
        self.vocab.update(tokens)
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        return tokens, load_embeddings(self.vocab)

    def generate(self, start_tokens, length=10, temperature=1.0):
        current_tokens = start_tokens.copy()
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} 
                 for token in current_tokens]
        for _ in range(length - len(start_tokens)):
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]
            if not next_token_probs: break
            adjusted_probs = {k: v ** (1/temperature) for k,v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            next_token = random.choices(list(adjusted_probs.keys()), 
                                      weights=[v/total for v in adjusted_probs.values()])[0]
            rationale = (
                f"Chose '{next_token}' because after '{current_tokens[-2]} {current_tokens[-1]}', "
                f"it appeared {next_token_probs[next_token]} times in training "
                f"(adjusted by temperature {temperature:.1f})"
            )
            result.append({"token": next_token, "rationale": rationale})
            current_tokens.append(next_token)
        return result

# =============================================================================
# Main App Content
# =============================================================================
def main():
    st.markdown('<div class="header"><h1>💻 Interactive LLM Explorer (UK English)</h1></div>', 
               unsafe_allow_html=True)
    
    # --------------------------
    # Guided Tutorial Section
    # --------------------------
    with st.expander("🎓 How This Works - Read Me First!", expanded=True):
        st.markdown("""
        **Learn Through Cooking Analogy:**  
        🧑🍳 Imagine training a chef assistant:
        1. **Training Phase** = Watching recipes being made  
        2. **Generation Phase** = Guessing next steps in a recipe  
        3. **Temperature** = Chef's creativity vs. strictness
        
        **Key Limitations:**  
        - Only remembers **pairs of ingredients** (not full recipes)  
        - Makes guesses based on what it's seen before  
        - Quality depends completely on training recipes provided  
        """)
    
    # --------------------------
    # Training Section
    # --------------------------
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    st.header("🍳 Step 1: Train Your Model")
    input_text = st.text_area("Enter training text (UK English):", 
                            height=150,
                            placeholder="E.g., 'My father’s family name being Pirrip, and my Christian name Philip...'")
    
    if st.button("Train Model", help="Click to process the training text"):
        if not input_text.strip():
            st.warning("Please enter training text first")
        else:
            st.session_state.language_model = SimpleLanguageModel()
            tokens, embeddings = st.session_state.language_model.train(input_text)
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings
            st.success(f"Trained on {len(tokens)} tokens! Try generating text below.")
            
            # Show top trigrams
            st.subheader("🔍 Top Learned Patterns")
            trigrams = [
                (f"{w1} {w2} → {next_word}", count)
                for (w1,w2), next_words in st.session_state.language_model.model.items()
                for next_word, count in next_words.items()
            ]
            df = pd.DataFrame(trigrams, columns=['Pattern', 'Count']).sort_values('Count', ascending=False)
            st.dataframe(df.head(10), use_container_width=True)
    
    # --------------------------
    # Generation Section
    # --------------------------
    st.header("🎭 Step 2: Generate Text")
    col1, col2 = st.columns([3,1])
    with col1:
        prompt = st.text_input("Enter your prompt (UK English):",
                              placeholder="E.g., 'My father’s'")
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 
                               help="Higher = more creative, Lower = more predictable")
    
    if st.button("Generate", type="primary"):
        if 'language_model' not in st.session_state:
            st.error("Please train the model first")
        else:
            prompt_tokens = tokenizer.tokenize(prompt.lower())
            if len(prompt_tokens) < 2:
                st.warning("Please enter at least two words")
            else:
                start_tokens = prompt_tokens[:2]
                output = st.session_state.language_model.generate(start_tokens, 
                                                                temperature=temperature,
                                                                length=20)
                generated = " ".join([entry["token"] for entry in output])
                
                # Display results
                st.subheader("Generated Text")
                st.markdown(f'<div class="explanation-box">{generated}</div>', 
                           unsafe_allow_html=True)
                
                # Debug console
                st.subheader("🛠️ Debug Console")
                with st.expander("See Generation Process"):
                    for entry in output[2:]:  # Skip first two tokens
                        st.write(f"**{entry['token']}** - {entry['rationale']}")
                
                # Training data context
                relevant_trigrams = st.session_state.language_model.model.get(tuple(start_tokens), {})
                if relevant_trigrams:
                    st.write(f"**Training Context for '{' '.join(start_tokens)}':**")
                    st.bar_chart(pd.Series(relevant_trigrams).sort_values(ascending=False).head(5))
                else:
                    st.error("No training examples for this context - try different prompt")
    
    # --------------------------
    # Educational Quizzes
    # --------------------------
    if 'language_model' in st.session_state:
        st.header("🧠 Knowledge Check")
        sample_trigrams = [
            (f"{w1} {w2} → {next_word}", count)
            for (w1,w2), next_words in st.session_state.language_model.model.items()
            for next_word, count in next_words.items()
        ]
        if sample_trigrams:
            selected = random.choice(sample_trigrams)
            st.write(f"**Quiz:** How many times did the model see this pattern?")
            st.code(f"{selected[0]}", language="text")
            
            col1, col2 = st.columns(2)
            with col1:
                user_guess = st.number_input("Your guess:", min_value=0)
            with col2:
                if st.button("Check Answer"):
                    st.write(f"**Answer:** {selected[1]} occurrences")
                    if user_guess == selected[1]:
                        st.success("Correct! 🎉")
                    else:
                        st.error(f"Try again! Actual count was {selected[1]}")
    
    # --------------------------
    # Footer
    # --------------------------
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="footer">
            <p>© 2024 Bloch AI LTD - All Rights Reserved<br>
            <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
