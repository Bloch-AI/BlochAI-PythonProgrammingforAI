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
# Custom CSS Styling for a Cleaner Look
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
            background-color: #333;
            color: #ccc;
            text-align: center;
            padding: 1rem;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions and Classes 
# =============================================================================

# Instantiate our simple tokenizer (using the TreebankWordTokenizer, which needs no extra data)
tokenizer = TreebankWordTokenizer()

def load_embeddings(vocab):
    """Generate random 50-dimensional embeddings for each word (for demonstration)."""
    return {word: np.random.rand(50) for word in vocab}

class SimpleLanguageModel:
    """
    A very basic trigram-based language model. It "learns" from a training text by counting 
    groups of three consecutive words (trigrams). Later, it uses this information to generate
    new text based on a starting two-word context.
    """
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))  # Maps (word1, word2) to counts of word3
        self.vocab = set()

    def train(self, text):
        """
        Tokenises the input text (converted to lower case) and builds the trigram counts.
        It also generates random embeddings for each unique word (for later visualisation).
        """
        tokens = tokenizer.tokenize(text.lower())
        self.vocab.update(tokens)
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        return tokens, load_embeddings(self.vocab)

    def generate(self, start_tokens, length=10, temperature=1.0):
        """
        Given a starting two-word context, generate additional tokens based on the trigram probabilities.
        The temperature parameter adjusts randomness:
          - Lower temperature → more predictable, high-frequency choices.
          - Higher temperature → more creative, random choices.
        **Note:** This model only uses the last two words to decide the next word.
        """
        current_tokens = start_tokens.copy()
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} for token in current_tokens]
        for _ in range(length - len(start_tokens)):
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]
            if not next_token_probs:
                break  # If this two-word context was never seen, stop generating.
            # Apply temperature adjustment to smooth or sharpen probabilities
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
    Simulate an 'attention' matrix (as seen in Transformer models) by generating random values.
    Each row is normalised to sum to 1.
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
    This is solely for demonstration to show how token embeddings might be transformed.
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
    Display a bar chart of the top 10 most common trigram patterns learned from the training text.
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
        st.info("No trigram patterns were found. Please provide more training text.")

def display_ngram_wordcloud(model):
    """
    Display a word cloud of the learned trigram patterns. Larger text indicates higher frequency.
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
        st.info("No word cloud to display. Please train the model with more data.")

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
        This app demonstrates a very simple language model that works by learning *trigrams* (groups of three words).  
        
        **Training Phase:**  
        - You provide some training text (for example, a passage from a book).  
        - The model tokenises the text and counts how often each pair of words is followed by a third word.  
        
        **Generation Phase:**  
        - You then enter a prompt (a question or statement).  
        - **Important:** The model only uses the **first two words** of your prompt as context.  
        - Based on this limited context, the model generates additional words.  
        
        **Limitations:**  
        - Because it only remembers two-word sequences, the model often cannot generate the full, correct answer.  
          For instance, if the training text contains the line:  
            "My father’s family name being **Pirrip**, and my Christian name Philip..."  
          and you ask "My father’s family name being?", the expected answer is "Pirrip".  
          However, the model only sees `["my", "father’s"]` and may output something different (e.g. "gave me ...") because in the training text “my father’s” was followed by other words more often.
        
        **Improving Results:**  
        - Provide more training text.  
        - Use a longer prompt to give more context.  
        - Understand that this trigram model is a very simplified demonstration compared to real LLMs.
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
                        'The model has learned patterns from your text by counting how often each pair of words is followed by another word.'
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
                               placeholder="E.g., 'My father’s'")
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0,
                                help="Higher values make the output more random (creative), lower values make it more predictable.")
    
    if st.button("Generate", type="primary"):
        if 'language_model' not in st.session_state:
            st.error("Please train the model first by entering training text and clicking 'Train Model'.")
        else:
            prompt_tokens = tokenizer.tokenize(prompt.lower())
            if len(prompt_tokens) < 2:
                st.warning("Please enter at least two words in your prompt.")
            else:
                start_tokens = prompt_tokens[:2]
                st.markdown(f"<div class='explanation-box'>Using starting context: <strong>{' '.join(start_tokens)}</strong></div>",
                            unsafe_allow_html=True)
                output = st.session_state.language_model.generate(start_tokens, length=20, temperature=temperature)
                generated = " ".join([entry["token"] for entry in output])
                
                st.subheader("Generated Text")
                st.markdown(f"<div class='explanation-box'>{generated}</div>", unsafe_allow_html=True)
                
                with st.expander("See How the Answer Was Generated (Step-by-Step)"):
                    for entry in output[2:]:  # Skip the first two tokens (the prompt context)
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
    # Footer Section
    # --------------------------
    st.markdown("""
        <div class="footer">
            <p>© 2024 Bloch AI LTD - All Rights Reserved<br>
            <a href="https://www.bloch.ai" style="color: #ccc;">www.bloch.ai</a></p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
