import streamlit as st
import time
import random
import numpy as np

# -----------------------------------------------------------------------------
# Use a simple rule-based tokenizer from NLTK that does not require extra downloads.
# This avoids the need for the Punkt data.
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization
from sklearn.decomposition import PCA

# =============================================================================
# Helper Functions and Classes
# =============================================================================

def load_embeddings(vocab):
    """
    For each unique word in the vocabulary, generate a random 50-dimensional vector.
    In a real LLM, these would be pre-trained embeddings.
    """
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.rand(50)
    return embeddings

class SimpleLanguageModel:
    """
    A very basic language model based on trigrams (groups of three words). It learns
    which word tends to follow a given pair of words from the training text.
    """
    def __init__(self):
        # For each two-word tuple, we count how often each third word follows.
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, text):
        """
        Tokenises the input text (converted to lower case) and builds a trigram model.
        """
        # Tokenise the text using our simple tokenizer.
        tokens = tokenizer.tokenize(text.lower())
        self.vocab.update(tokens)
        # Build the trigram counts.
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        # Generate random embeddings for demonstration.
        embeddings = load_embeddings(self.vocab)
        return tokens, embeddings

    def generate(self, start_tokens, length=10, temperature=1.0):
        """
        Starting with a two-word context, generate additional tokens based on the
        learned trigram probabilities. The 'temperature' parameter adjusts randomness.
        """
        current_tokens = start_tokens.copy()
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} for token in current_tokens]
        for _ in range(length - len(start_tokens)):
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]
            if not next_token_probs:
                # If the current two-word context was never seen in training, stop generation.
                break
            # Adjust probabilities by temperature.
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
            # Randomly choose the next token.
            next_token = random.choices(list(adjusted_probs.keys()), weights=list(adjusted_probs.values()))[0]
            rationale = (
                f"Token '{next_token}' was chosen because, in the training text, it followed "
                f"the sequence '{current_tokens[-2]} {current_tokens[-1]}'. "
                f"(Original frequencies: {next_token_probs}, adjusted for temperature: {adjusted_probs})"
            )
            result.append({"token": next_token, "rationale": rationale})
            current_tokens.append(next_token)
        return result

def simulate_attention(input_tokens, output_tokens):
    """
    Simulate an attention matrix (as used in Transformer models) using random values.
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
    A simplified Transformer block that applies multi-head self-attention and a feed-forward
    network. This is only for illustration purposes.
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
    Show a bar chart of the 10 most common trigrams (three-word sequences) learned from the training text.
    """
    st.subheader("Step 3: Trigram Frequency Visualisation")
    st.write("The bar chart below shows the 10 most common three-word sequences in your training text.")
    ngram_counts = []
    for (w1, w2), next_words in model.items():
        for next_word, count in next_words.items():
            ngram_counts.append((f"{w1} {w2} {next_word}", count))
    if ngram_counts:
        df = pd.DataFrame(ngram_counts, columns=['Trigram', 'Count'])
        df = df.sort_values(by='Count', ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Count', y='Trigram', data=df)
        plt.title('Top 10 Trigrams by Frequency')
        st.pyplot(plt)
        plt.clf()
    else:
        st.write("No trigrams were found. Please enter more training text.")

def display_ngram_wordcloud(model):
    """
    Show a word cloud of the trigrams learned from the training text. Larger text indicates higher frequency.
    """
    st.subheader("Step 4: Trigram Word Cloud")
    st.write("The word cloud below shows the trigrams learned from your training text. Bigger words indicate a higher frequency.")
    ngram_counts = defaultdict(int)
    for (w1, w2), next_words in model.items():
        for next_word, count in next_words.items():
            ngram_counts[f"{w1} {w2} {next_word}"] += count
    if ngram_counts:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        st.pyplot(plt)
        plt.clf()
    else:
        st.write("No trigrams were found. Please enter more training text.")

# =============================================================================
# Main Function for the Streamlit App
# =============================================================================

def main():
    st.title("💻 LLM Simulator App (UK English)")
    st.write(
        """
        **Welcome!**  
        This educational app demonstrates, in a simplified way, how a large language model (LLM) works.  
        
        **Step 1:** Enter training text.  
        **Step 2:** The model learns patterns from your text (using groups of three words).  
        **Step 3:** You can view visualisations of what the model has learnt.  
        **Step 4:** Ask a question (prompt) and the model will generate an answer based on its learning.
        
        *Note:* In real life, LLMs are very complex. This demo uses a simple trigram model for clarity.
        """
    )

    # --------------------------------------------------------------------------
    # STEP 1: Input Training Text
    # --------------------------------------------------------------------------
    st.header("Step 1: Enter Training Text")
    st.write("Please enter some text below. The model will use this text to learn word patterns (in UK English).")
    input_text = st.text_area("Training Text", height=150)

    if st.button("Train Model"):
        if input_text.strip() == "":
            st.warning("Please enter some training text to proceed.")
        else:
            # Initialise the language model.
            if 'language_model' not in st.session_state:
                st.session_state.language_model = SimpleLanguageModel()
            tokens, embeddings = st.session_state.language_model.train(input_text)
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings

            st.success("Training complete! The model has learned from your text.")
            st.write("For example, it has learnt these words and trigram patterns:")
            model_display = {str(k): dict(v) for k, v in st.session_state.language_model.model.items()}
            st.json(model_display)

            # Display visualisations.
            display_ngram_bar_chart(st.session_state.language_model.model)
            display_ngram_wordcloud(st.session_state.language_model.model)

    # --------------------------------------------------------------------------
    # STEP 2: Ask a Question (Prompt)
    # --------------------------------------------------------------------------
    st.header("Step 2: Ask a Question")
    st.write(
        "LLMs work by taking a prompt (a question or instruction) and generating an answer based on what they have learned. "
        "Before asking a question, please ensure you have trained the model above."
    )
    prompt_text = st.text_input("Your Question", placeholder="Enter your question here (in UK English)")
    if st.button("Get Answer"):
        # Check if the model has been trained.
        if 'language_model' not in st.session_state or 'tokens' not in st.session_state:
            st.error("Please train the model first by entering training text and clicking 'Train Model'.")
        elif prompt_text.strip() == "":
            st.warning("Please enter a question.")
        else:
            prompt_tokens = tokenizer.tokenize(prompt_text.lower())
            if len(prompt_tokens) < 2:
                st.warning("Please enter a longer question (at least two words).")
            else:
                # Use the first two tokens of the prompt as the starting context.
                start_tokens = prompt_tokens[:2]
                st.write(f"Using the starting context: {start_tokens}")
                # Generate an answer. (Here we attempt to generate 20 tokens.)
                output_with_rationales = st.session_state.language_model.generate(start_tokens, length=20, temperature=1.0)
                generated_tokens = [entry["token"] for entry in output_with_rationales]
                # If no additional tokens were generated, explain why.
                if len(generated_tokens) <= 2:
                    st.error("The model could not generate an answer. "
                             "This may be because the training text does not provide enough context for the given prompt. "
                             "Please try training the model with more detailed text.")
                else:
                    answer = " ".join(generated_tokens)
                    st.subheader("Answer")
                    st.write(answer)
                    with st.expander("See How the Answer Was Generated (Step-by-Step)"):
                        for i, entry in enumerate(output_with_rationales):
                            st.write(f"{i}: **{entry['token']}** — {entry['rationale']}")

    # --------------------------------------------------------------------------
    # Advanced Visualisations (Optional)
    # --------------------------------------------------------------------------
    if 'output_with_rationales' in st.session_state:
        st.header("Additional Visualisations (Advanced)")
        st.subheader("Attention Visualisation")
        st.write(
            "The heatmap below simulates how each generated word 'attends' to the training text. "
            "Darker colours indicate stronger connections."
        )
        attention_matrix = simulate_attention(st.session_state.tokens, st.session_state.generated_tokens)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix,
                    xticklabels=st.session_state.tokens,
                    yticklabels=st.session_state.generated_tokens,
                    ax=ax, cmap="YlOrRd")
        plt.xlabel("Training Text Tokens")
        plt.ylabel("Generated Answer Tokens")
        st.pyplot(plt)
        plt.clf()

        st.subheader("Transformer Block Simulation")
        st.write(
            "A Transformer block processes word embeddings to produce new representations. "
            "Below, PCA is used to reduce the dimensions of the original and transformed embeddings so you can see the difference."
        )
        tokens = st.session_state.tokens
        embeddings = st.session_state.embeddings
        embedding_matrix = np.array([embeddings[token] for token in tokens])
        # Add a batch dimension (required by the Transformer block).
        embedding_matrix_batch = np.expand_dims(embedding_matrix, axis=0)
        transformer_block = SimpleTransformerBlock(embed_dim=50, num_heads=2, ff_dim=64)
        transformed_embeddings = transformer_block(embedding_matrix_batch, training=False)
        transformed_embeddings = transformed_embeddings.numpy().squeeze(0)
        # Combine original and transformed embeddings for PCA.
        combined = np.concatenate([embedding_matrix, transformed_embeddings], axis=0)
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        original_2d = combined_2d[:len(tokens)]
        transformed_2d = combined_2d[len(tokens):]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(original_2d[:, 0], original_2d[:, 1], color='blue', label='Original Embeddings')
        for i, token in enumerate(tokens):
            ax.annotate(token, (original_2d[i, 0], original_2d[i, 1]), color='blue', fontsize=8)
        ax.scatter(transformed_2d[:, 0], transformed_2d[:, 1], color='red', label='Transformed Embeddings')
        for i, token in enumerate(tokens):
            ax.annotate(token, (transformed_2d[i, 0], transformed_2d[i, 1]), color='red', fontsize=8)
        ax.legend()
        ax.set_title("PCA of Token Embeddings: Original (blue) vs Transformed (red)")
        st.pyplot(plt)
        plt.clf()

if __name__ == "__main__":
    main()

# =============================================================================
# Footer
# =============================================================================
footer = st.container()
footer.markdown(
    '''
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: black;
        color: white;
        text-align: center;
        padding: 10px 0;
    }
    </style>
    <div class="footer">
        <p>© 2024 Bloch AI LTD - All Rights Reserved. <a href="https://www.bloch.ai" style="color: white;">www.bloch.ai</a></p>
    </div>
    ''',
    unsafe_allow_html=True
)
