import streamlit as st  # Importing Streamlit for building the web app
import time  # Importing time for potential future use (not used here)
import random  # Importing random for generating random numbers and choices
import numpy as np  # Importing numpy for numerical operations
import nltk  # Importing nltk for natural language processing tasks
from nltk.tokenize import word_tokenize  # To split text into words
from collections import defaultdict  # For dictionaries with default values
import matplotlib.pyplot as plt  # For plotting charts and graphs
import seaborn as sns  # For enhancing plot styles
from wordcloud import WordCloud  # For generating word cloud visualizations
import pandas as pd  # For data manipulation
import tensorflow as tf  # For deep learning operations
from tensorflow.keras.layers import Dense, LayerNormalization  # For building neural network layers
from sklearn.decomposition import PCA  # For dimensionality reduction in visualization

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Function to load (or simulate) pre-trained word embeddings.
def load_embeddings(vocab):
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.rand(50)  # 50-dimensional random vector
    return embeddings

# A simple trigram language model.
class SimpleLanguageModel:
    def __init__(self):
        # The model uses a nested defaultdict to count the occurrences of each trigram.
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()  # The vocabulary of unique words

    def train(self, text):
        # Tokenize the text into lowercase words.
        tokens = word_tokenize(text.lower())
        self.vocab.update(tokens)
        # Build trigrams from the tokens.
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        # Create embeddings for all unique words.
        embeddings = load_embeddings(self.vocab)
        return tokens, embeddings

    def generate(self, start_tokens, length=10, temperature=1.0):
        current_tokens = start_tokens.copy()
        # Record the starting tokens with a simple rationale.
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} for token in current_tokens]
        # Generate new tokens until reaching the desired length.
        for _ in range(length - len(start_tokens)):
            # Get next-token probabilities using the last two tokens as context.
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]
            if not next_token_probs:
                break  # No next token found for this context.
            # Adjust probabilities with temperature (higher temperature = more randomness).
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
            # Randomly choose the next token based on adjusted probabilities.
            next_token = random.choices(list(adjusted_probs.keys()), weights=list(adjusted_probs.values()))[0]
            rationale = (
                f"Token: '{next_token}' selected. "
                f"Original probabilities: {next_token_probs}. "
                f"Adjusted probabilities (temperature={temperature}): {adjusted_probs}. "
                f"Temperature of {temperature} {'flattened' if temperature > 1 else 'sharpened' if temperature < 1 else 'maintained'} the probability distribution."
            )
            result.append({"token": next_token, "rationale": rationale})
            current_tokens.append(next_token)
        return result

# Function to simulate an attention matrix between input and output tokens.
def simulate_attention(input_tokens, output_tokens):
    attention_matrix = [
        [random.random() for _ in range(len(input_tokens))]
        for _ in range(len(output_tokens))
    ]
    # Normalize each row to sum to 1.
    for row in attention_matrix:
        total = sum(row)
        for i in range(len(row)):
            row[i] /= total
    return attention_matrix

# A custom, simplified Transformer block.
class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(SimpleTransformerBlock, self).__init__()
        # Multi-head self-attention layer.
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        # Feed-forward network with one hidden layer.
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

# Display the top 10 trigrams in a bar chart.
def display_ngram_bar_chart(model):
    st.subheader("N-gram Bar Chart")
    st.write("The bar chart below shows the frequency of each trigram (three-word sequence) learned from the input text.")
    
    ngram_counts = []
    for (w1, w2), next_words in model.items():
        for next_word, count in next_words.items():
            ngram_counts.append((f"{w1} {w2} {next_word}", count))
    
    if ngram_counts:
        df = pd.DataFrame(ngram_counts, columns=['N-gram', 'Count'])
        df = df.sort_values(by='Count', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Count', y='N-gram', data=df)
        plt.title('Top 10 N-grams by Frequency')
        st.pyplot(plt)
        plt.clf()
    else:
        st.write("No n-grams were learned. Please input more text.")

# Display a word cloud for the learned trigrams.
def display_ngram_wordcloud(model):
    st.subheader("N-gram Word Cloud")
    st.write("The word cloud visualizes the n-grams learned from the input text. Larger sizes indicate higher frequency.")
    
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
        st.write("No n-grams were learned. Please input more text.")

# Main function to run the Streamlit app.
def main():
    st.title("💻 LLM Simulator App")
    st.write(
        """
        This app simulates the workings of a large language model (LLM) by:
        
        - **Training** on your input text (using a simple trigram model).
        - **Generating** text with a rationale for each choice.
        - **Visualizing** token-level attention.
        - **Displaying** next-token probability distributions.
        - **Simulating** a Transformer block to show how token embeddings change.
        
        Enjoy exploring the internals of an LLM!
        """
    )

    # Initialize the language model in session state.
    if 'language_model' not in st.session_state:
        st.session_state.language_model = SimpleLanguageModel()

    # Text area for user input.
    input_text = st.text_area("Enter your text here:", height=100)

    # Slider inputs for model parameters.
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    with col2:
        output_length = st.slider("Output Length", 5, 50, 20)

    # Process the input text when the button is pressed.
    if st.button("Process"):
        if input_text:
            # Train the model and obtain tokens and embeddings.
            tokens, embeddings = st.session_state.language_model.train(input_text)
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings

            # For generation, use the first two tokens as the starting context.
            if len(tokens) < 2:
                st.error("Please enter more text to form a valid context.")
            else:
                start_tokens = tokens[:2]
                output_with_rationales = st.session_state.language_model.generate(start_tokens, output_length, temperature)
                st.session_state.output_with_rationales = output_with_rationales
                st.session_state.generated_tokens = [entry["token"] for entry in output_with_rationales]
                st.session_state.generated_sentence = " ".join(st.session_state.generated_tokens)
                # For debugging/education, display the underlying model counts.
                model_display = {str(k): dict(v) for k, v in st.session_state.language_model.model.items()}
                st.write("Learned trigram model:", model_display)
        else:
            st.warning("Please enter some text to process.")

    # If generation is complete, show outputs and visualizations.
    if 'output_with_rationales' in st.session_state:
        st.subheader("LLM Generated Output")
        st.write("**Generated Sentence:**", st.session_state.generated_sentence)
        
        with st.expander("Output Tokens with Rationales", expanded=True):
            for i, entry in enumerate(st.session_state.output_with_rationales):
                st.write(f"{i}: **{entry['token']}** — {entry['rationale']}")

        # Attention Visualization: How each generated token "attends" to the input tokens.
        st.subheader("Attention Visualization")
        st.write("This heatmap shows simulated attention scores between each output token and each input token.")
        attention_matrix = simulate_attention(st.session_state.tokens, st.session_state.generated_tokens)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix, 
                    xticklabels=st.session_state.tokens, 
                    yticklabels=st.session_state.generated_tokens, 
                    ax=ax, cmap="YlOrRd")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        st.pyplot(plt)
        plt.clf()

        # Token Probability Visualization:
        st.subheader("Token Probability Visualization")
        st.write(
            "Select a context (two consecutive tokens) from the model to view the original and temperature-adjusted "
            "probability distributions for the next token."
        )
        contexts = list(st.session_state.language_model.model.keys())
        if contexts:
            # Create a string representation for each context.
            context_options = [' '.join(context) for context in contexts]
            selected_context_str = st.selectbox("Select a two-token context:", context_options)
            selected_context = tuple(selected_context_str.split())
            st.write("**Selected context:**", selected_context_str)
            next_token_probs = st.session_state.language_model.model[selected_context]
            # Adjust probabilities based on the temperature.
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

            if next_token_probs:
                fig, ax = plt.subplots(figsize=(10, 6))
                # Show the top 10 probabilities.
                sorted_probs = sorted(next_token_probs.items(), key=lambda x: x[1], reverse=True)[:10]
                tokens_list, probs = zip(*sorted_probs)
                sorted_adj = sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)[:10]
                adj_tokens, adj_probs = zip(*sorted_adj)
                bar_width = 0.35
                index = np.arange(len(tokens_list))
                ax.bar(index, probs, bar_width, label='Original Probabilities')
                ax.bar(index + bar_width, adj_probs, bar_width, label='Adjusted Probabilities')
                plt.xticks(index + bar_width / 2, tokens_list, rotation=45, ha='right')
                plt.xlabel("Next Tokens")
                plt.ylabel("Probability")
                plt.legend()
                st.pyplot(plt)
                plt.clf()
            else:
                st.write("No next token probabilities available for the selected context.")
        else:
            st.write("No two-token contexts available. Please input more text.")

        # Transformer Block Simulation:
        if 'embeddings' in st.session_state and 'tokens' in st.session_state and len(st.session_state.tokens) > 0:
            st.subheader("Transformer Block Simulation")
            st.write(
                "A simple Transformer block processes token embeddings to produce new representations. "
                "Below we use PCA to reduce both the original and transformed embeddings to 2 dimensions for visualization."
            )
            tokens = st.session_state.tokens
            embeddings = st.session_state.embeddings
            # Build an embedding matrix for the input tokens (in order).
            embedding_matrix = np.array([embeddings[token] for token in tokens])
            # Add a batch dimension for the Transformer block: shape (1, seq_len, embed_dim)
            embedding_matrix_batch = np.expand_dims(embedding_matrix, axis=0)
            # Create and apply a simple Transformer block.
            transformer_block = SimpleTransformerBlock(embed_dim=50, num_heads=2, ff_dim=64)
            transformed_embeddings = transformer_block(embedding_matrix_batch, training=False)
            # Remove the batch dimension.
            transformed_embeddings = transformed_embeddings.numpy().squeeze(0)
            # Combine original and transformed embeddings to compute a common PCA projection.
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

# Run the main function to start the app.
if __name__ == "__main__":
    main()

# Footer at the bottom of the app.
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
