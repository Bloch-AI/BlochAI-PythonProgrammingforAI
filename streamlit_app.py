import streamlit as st
import random
import numpy as np
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization
from sklearn.decomposition import PCA

# =============================================================================
# Custom CSS Styling for a Cleaner Look and Full-Width Footer
# =============================================================================
st.markdown("""
    <style>
        body {
            background-color: #ffffff;
            font-family: "Segoe UI", sans-serif;
        }
        .header {
            background-color: #4a90e2;
            color: #ffffff;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .explanation-box {
            background-color: #e1f5fe;
            border-left: 4px solid #0288d1;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        .debug-console {
            background-color: #fff9c4;
            border: 1px solid #fbc02d;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #333;
            color: #ccc;
            text-align: center;
            padding: 1rem 0;
            font-size: 0.9rem;
            z-index: 100;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar: Detailed Explanation of How an LLM Works
# =============================================================================
st.sidebar.markdown("""
# How Does an LLM Work?

**Large Language Models (LLMs)** are sophisticated systems trained on huge amounts of text.  
They learn statistical patterns and relationships between words to generate text that is coherent and contextually relevant.

### In This Demo:
- **Simplified Model:**  
  Our model is based on *trigrams* (groups of 3 words). It only remembers which word usually follows a given pair of words.

- **Training Phase:**  
  You provide training text (e.g., a passage from a book). The model tokenises this text and counts how often each two-word sequence is followed by another word.

- **Generation Phase:**  
  You enter a prompt.  
  **Important:** This demo uses the **first five words** of your prompt as the starting context.  
  The model then generates additional words based on its limited training data.

- **Limitations:**  
  - If the training text is too short or not relevant, the model cannot generate a good answer.  
  - It only uses a very short context (5 words), unlike real LLMs which use dozens or hundreds of tokens.  
  - More training data and a longer prompt can lead to better results.

Remember: This demo is for educational purposes only and is a very simplified representation of how real LLMs operate.
""")

# =============================================================================
# Helper Functions and Classes 
# =============================================================================

# Instantiate our tokenizer (TreebankWordTokenizer requires no extra data)
tokenizer = TreebankWordTokenizer()

def load_embeddings(vocab):
    """Generate random 50-dimensional embeddings for each word (for demonstration)."""
    return {word: np.random.rand(50) for word in vocab}

class SimpleLanguageModel:
    """
    A basic trigram-based language model. It learns by counting how often two consecutive words
    are followed by a third word in the training text.
    """
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))  # Maps (word1, word2) → counts of word3
        self.vocab = set()

    def train(self, text):
        """
        Tokenises the input text (converted to lower case) and builds the trigram model.
        Also generates random embeddings for each unique word (for later visualisation).
        """
        tokens = tokenizer.tokenize(text.lower())
        self.vocab.update(tokens)
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        return tokens, load_embeddings(self.vocab)

    def generate(self, start_tokens, length=10, temperature=1.0):
        """
        Generates text starting from a given context (start_tokens).
        This model uses only the last two words of the context to predict the next word.
        The 'temperature' parameter controls randomness:
          - Lower temperature: more predictable, high-frequency words.
          - Higher temperature: more creative, varied output.
        """
        current_tokens = start_tokens.copy()
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} for token in current_tokens]
        for _ in range(length - len(start_tokens)):
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]
            if not next_token_probs:
                break  # Stop if this context was never seen
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
            next_token = random.choices(list(adjusted_probs.keys()),
                                        weights=list(adjusted_probs.values()))[0]
            rationale = (
                f"Chose '{next_token}' because after '{current_tokens[-2]} {current_tokens[-1]}' "
                f"it appeared {next_token_probs[next_token]} times (temperature-adjusted)."
            )
            result.append({"token": next_token, "rationale": rationale})
            current_tokens.append(next_token)
        return result

def simulate_attention(input_tokens, output_tokens):
    """
    Simulate an 'attention' matrix (like in Transformer models) with random values.
    Each row is normalised so that the scores add up to 1.
    """
    attention_matrix = [
        [random.random() for _ in range(len(input_tokens))]
        for _ in range(len(output_tokens))
    ]
    for row in attention_matrix:
        total = sum(row)
        for i in range(len(row)):
            row[i] /= total
    return attention_matrix

class SimpleTransformerBlock(tf.keras.layers.Layer):
    """
    A simplified Transformer block with multi-head self-attention and a feed-forward network.
    This block is only for visual demonstration of how token embeddings can be transformed.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(SimpleTransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def display_ngram_bar_chart(model):
    """
    Display a table of the top 10 most common trigram patterns learned from the training text.
    """
    st.subheader("Learned Trigram Patterns")
    patterns = [
        (f"{w1} {w2} → {w3}", count)
        for (w1, w2), next_words in model.items()
        for w3, count in next_words.items()
    ]
    if patterns:
        df = pd.DataFrame(patterns, columns=['Pattern', 'Count']).sort_values('Count', ascending=False)
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("No trigram patterns found. Please provide more training text.")

def display_ngram_wordcloud(model):
    """
    Display a word cloud of the learned trigram patterns.
    Larger words indicate higher frequency.
    """
    st.subheader("Trigram Word Cloud")
    ngram_counts = defaultdict(int)
    for (w1, w2), next_words in model.items():
        for w3, count in next_words.items():
            ngram_counts[f"{w1} {w2} {w3}"] += count
    if ngram_counts:
        wordcloud = WordCloud(width=800, height=400, background_color='white') \
                        .generate_from_frequencies(ngram_counts)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)
        plt.clf()
    else:
        st.info("No word cloud to display. Train the model with more data.")

# =============================================================================
# Main App Content
# =============================================================================
def main():
    # Header
    st.markdown('<div class="header"><h1>Interactive LLM Explorer (UK English)</h1></div>',
                unsafe_allow_html=True)

    # --------------------------
    # Guided Tutorial / Explanation
    # --------------------------
    with st.expander("How This App Works (Read Me First)", expanded=True):
        st.markdown("""
        **Overview:**  
        This app demonstrates a very simple language model that learns *trigrams* (groups of three words).  

        **Training Phase:**  
        - You enter training text (e.g., an excerpt from a book).  
        - The app tokenises the text and counts how often each pair of words is followed by another word.

        **Generation Phase:**  
        - You enter a prompt (a question or statement).  
        - **Important:** The model uses the first **five words** of your prompt as its starting context.  
        - It then generates additional words based solely on this limited context.

        **Key Limitations:**  
        - The model only remembers two-word contexts, so it may not produce the "correct" answer.  
          For example, if your training text contains:  
          *"My father’s family name being Pirrip, and my Christian name Philip..."*  
          and you prompt: *"My father’s family name being?"*,  
          the expected answer is **Pirrip**. However, if the model rarely saw the full sequence leading to "Pirrip" (because it only uses the first five words of the prompt), it might generate something else.  
        - Results improve with more training data and more context in the prompt.
        """)
    
    # --------------------------
    # Step 1: Training
    # --------------------------
    st.header("Step 1: Train Your Model")
    input_text = st.text_area("Enter your training text (UK English):", 
                              height=150,
                              placeholder="E.g., 'My father’s family name being Pirrip, and my Christian name Philip...'")
    
    if st.button("Train Model", help="Click here after entering your training text"):
        if not input_text.strip():
            st.warning("Please enter some training text to proceed.")
        else:
            st.session_state.language_model = SimpleLanguageModel()
            tokens, embeddings = st.session_state.language_model.train(input_text)
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings
            st.success(f"Training complete! Model processed {len(tokens)} tokens.")
            
            st.markdown('<div class="explanation-box">'
                        'The model has learned patterns by counting how often each pair of words is followed by a third word.'
                        '</div>', unsafe_allow_html=True)
            
            st.subheader("Learned Trigram Patterns")
            display_ngram_bar_chart(st.session_state.language_model.model)
            display_ngram_wordcloud(st.session_state.language_model.model)

    # --------------------------
    # Step 2: Generation
    # --------------------------
    st.header("Step 2: Generate Text")
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input("Enter your prompt (UK English):",
                               placeholder="E.g., 'My father’s family name being'")
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0,
                                help="Higher values increase randomness (more creative), lower values are more predictable.")
    
    if st.button("Generate", type="primary"):
        if 'language_model' not in st.session_state:
            st.error("Please train the model first by entering training text and clicking 'Train Model'.")
        else:
            prompt_tokens = tokenizer.tokenize(prompt.lower())
            if len(prompt_tokens) < 5:
                st.warning("Please enter at least five words in your prompt for better context.")
            else:
                # Use the first five words of the prompt as the starting context
                start_tokens = prompt_tokens[:5]
                st.markdown(f"<div class='explanation-box'>Using starting context: <strong>{' '.join(start_tokens)}</strong></div>",
                            unsafe_allow_html=True)
                output = st.session_state.language_model.generate(start_tokens, length=20, temperature=temperature)
                generated = " ".join([entry["token"] for entry in output])
                
                st.subheader("Generated Text")
                st.markdown(f"<div class='explanation-box'>{generated}</div>", unsafe_allow_html=True)
                
                with st.expander("See How the Answer Was Generated (Step-by-Step)"):
                    for entry in output[5:]:  # Skip the first five tokens (the prompt context)
                        st.write(f"**{entry['token']}** — {entry['rationale']}")
                
                # Show a simple bar chart for the current context from training data
                relevant_trigrams = st.session_state.language_model.model.get(tuple(start_tokens), {})
                if relevant_trigrams:
                    st.markdown(f"<div class='explanation-box'>Training examples for the context "
                                f"<strong>{' '.join(start_tokens)}</strong>:</div>",
                                unsafe_allow_html=True)
                    st.bar_chart(pd.Series(relevant_trigrams).sort_values(ascending=False).head(5))
                else:
                    st.info("No training examples found for this context. Try using a different prompt or add more training text.")

    # --------------------------
    # Full-Width Footer (Displayed Across the Entire Screen)
    # --------------------------
    st.markdown("""
        <div class="footer">
            <p>© 2024 Bloch AI LTD - All Rights Reserved | 
            <a href="https://www.bloch.ai" style="color: #ccc;">www.bloch.ai</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
