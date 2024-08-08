import streamlit as st  # Importing Streamlit for building the web app
import time  # Importing time for potential future use (not used here)
import random  # Importing random for generating random numbers and choices
import numpy as np  # Importing numpy for numerical operations
import nltk  # Importing nltk for natural language processing tasks
from nltk.tokenize import word_tokenize  # Importing word_tokenize to split text into words
from collections import defaultdict  # Importing defaultdict to handle dictionary with default values
import matplotlib.pyplot as plt  # Importing matplotlib for creating plots
import seaborn as sns  # Importing seaborn for enhancing the style of the plots
from wordcloud import WordCloud  # Importing WordCloud for generating word cloud visualizations
import pandas as pd  # Importing pandas for data manipulation
import tensorflow as tf  # Importing TensorFlow for deep learning operations
from tensorflow.keras.layers import Dense, LayerNormalization  # Importing layers for building a neural network

# Download necessary NLTK data
nltk.download('punkt', quiet=True)  # Downloading the Punkt tokenizer data used by word_tokenize

# Function to load pre-trained word embeddings, for simplicity using random embeddings here
def load_embeddings(vocab):
    embeddings = {}
    for word in vocab:
        embeddings[word] = np.random.rand(50)  # Generating 50-dimensional random embeddings for each word
    return embeddings

# Class for a simple language model using trigrams (three-word sequences)
class SimpleLanguageModel:
    def __init__(self):
        # Initializing the model with a nested defaultdict structure to count word occurrences
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()  # Set to store unique words in the vocabulary

    def train(self, text):
        # Tokenize the input text into lowercase words
        tokens = word_tokenize(text.lower())
        self.vocab.update(tokens)  # Update the vocabulary with tokens
        # Create trigrams and update model counts
        for i in range(len(tokens) - 2):
            self.model[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1
        # Generate embeddings for the vocabulary
        embeddings = load_embeddings(self.vocab)
        return tokens, embeddings

    def generate(self, start_tokens, length=10, temperature=1.0):
        current_tokens = start_tokens  # Start with the given start tokens
        result = [{"token": token, "rationale": f"Starting token: '{token}'"} for token in current_tokens]  # Store rationale for each token
        # Generate the rest of the sequence
        for _ in range(length - len(start_tokens)):
            next_token_probs = self.model[(current_tokens[-2], current_tokens[-1])]  # Get next word probabilities based on last two tokens
            if not next_token_probs:
                break  # If no probabilities, break the loop
            # Adjust probabilities with temperature (higher temperature = more randomness)
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}
            # Select the next token based on adjusted probabilities
            next_token = random.choices(list(adjusted_probs.keys()), 
                                        weights=list(adjusted_probs.values()))[0]
            # Store the rationale for the selection of the token
            rationale = (
                f"Token: '{next_token}' selected. "
                f"Original probabilities: {next_token_probs}. "
                f"Adjusted probabilities (temperature={temperature}): {adjusted_probs}. "
                f"Temperature of {temperature} {'flattened' if temperature > 1 else 'sharpened' if temperature < 1 else 'maintained'} the probability distribution."
            )
            result.append({"token": next_token, "rationale": rationale})  # Add the selected token and rationale to the result
            current_tokens.append(next_token)  # Update the current tokens with the selected token
        return result

# Function to simulate attention between tokens (randomized for demonstration purposes)
def simulate_attention(tokens, output_tokens):
    attention_matrix = [[random.random() for _ in range(len(tokens))] for _ in range(len(output_tokens))]  # Create a random attention matrix
    for row in attention_matrix:
        total = sum(row)
        for i in range(len(row)):
            row[i] /= total  # Normalize each row so that the sum of the attention scores is 1
    return attention_matrix

# Custom implementation of a simple Transformer block
class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(SimpleTransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)  # Multi-head attention layer
        # Feed-forward network with ReLU activation followed by a linear transformation
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)  # Layer normalization for stable training
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)  # Dropout for regularization
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        # Apply attention to the inputs
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)  # Apply dropout
        out1 = self.layernorm1(inputs + attn_output)  # Add residual connection and normalize
        # Pass through feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)  # Apply dropout
        return self.layernorm2(out1 + ffn_output)  # Add residual connection and normalize

# Function to display a bar chart of the top 10 most frequent trigrams (n-grams)
def display_ngram_bar_chart(model):
    st.subheader("N-gram Bar Chart")
    st.write("The bar chart below shows the frequency of each trigram in the input text.")

    ngram_counts = []
    for (w1, w2), next_words in model.items():
        for next_word, count in next_words.items():
            ngram_counts.append((f"{w1} {w2} {next_word}", count))  # Create a list of trigrams and their counts
    
    if ngram_counts:
        df = pd.DataFrame(ngram_counts, columns=['N-gram', 'Count'])  # Convert to DataFrame for easier manipulation
        df = df.sort_values(by='Count', ascending=False).head(10)  # Sort and take the top 10 trigrams
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Count', y='N-gram', data=df)  # Create a bar plot using seaborn
        plt.title('Top 10 N-grams by Frequency')
        st.pyplot(plt)  # Display the plot in the Streamlit app
    else:
        st.write("No n-grams were learned. Please input more text.")  # Handle case with no n-grams

# Function to display a word cloud of the learned trigrams
def display_ngram_wordcloud(model):
    st.subheader("N-gram Word Cloud")
    st.write("The word cloud below visualizes the n-grams learned from the input text. The size of each n-gram represents its frequency.")

    ngram_counts = defaultdict(int)
    for (w1, w2), next_words in model.items():
        for next_word, count in next_words.items():
            ngram_counts[f"{w1} {w2} {next_word}"] += count  # Aggregate counts for the word cloud
    
    if ngram_counts:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(ngram_counts)  # Generate word cloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")  # Display the word cloud
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("No n-grams were learned. Please input more text.")  # Handle case with no n-grams

# Main function to define the Streamlit app
def main():
    st.title("💻 LLM Simulator App")

    # Initialize the language model on the first run
    if 'language_model' not in st.session_state:
        st.session_state.language_model = SimpleLanguageModel()

    # Text area for user input
    input_text = st.text_area("Enter your text here:", height=100)

    # Slider inputs for model parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)  # Slider to adjust temperature
    with col2:
        output_length = st.slider("Output Length", 5, 50, 20)  # Slider to set the length of the generated output

    # Process the input text when the button is pressed
    if st.button("Process"):
        if input_text:
            # Train the model with the input text
            tokens, embeddings = st.session_state.language_model.train(input_text)
            
            # Store the tokens and generated data in session state to avoid recomputation
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings
            start_tokens = tokens[:2]  # Starting with the first two tokens
            # Generate output based on the trained model
            output_with_rationales = st.session_state.language_model.generate(start_tokens, output_length, temperature)
            st.session_state.output_with_rationales = output_with_rationales
            st.session_state.generated_tokens = [entry["token"] for entry in output_with_rationales]
            st.session_state.generated_sentence = " ".join(st.session_state.generated_tokens)

            # Convert tuple keys to strings for display
            model_display = {str(k): v for k, v in st.session_state.language_model.model.items()}
            st.write("Model contents:", model_display)  # Display the model's learned data
        
    if 'output_with_rationales' in st.session_state:
        st.subheader("LLM Output")
        st.write("Generated Output Sentence:", st.session_state.generated_sentence)  # Display the generated sentence

        with st.expander("Output Tokens with Rationales", expanded=True):
            for i, entry in enumerate(st.session_state.output_with_rationales):
                st.write(f"{i}: {entry['token']} - {entry['rationale']}")  # Display the generated tokens with their rationales

        st.subheader("Attention Visualization")
        st.write("The attention visualization shows how each output token attends to each input token.")
        attention_matrix = simulate_attention(st.session_state.tokens, st.session_state.generated_tokens)  # Simulate the attention matrix
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix, xticklabels=st.session_state.tokens, yticklabels=st.session_state.generated_tokens, ax=ax, cmap="YlOrRd")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        st.pyplot(plt)  # Display the attention heatmap

        st.subheader("Token Probability Visualization")
        st.write("The token probability visualization shows the likelihood of each possible next token given the current token, based on the training data.")
        
        if 'selected_token' not in st.session_state:
            st.session_state.selected_token = None
        
        # Dropdown to select a token and see its next token probabilities
        st.session_state.selected_token = st.selectbox("Select a token to see next token probabilities:", list(st.session_state.language_model.vocab), index=list(st.session_state.language_model.vocab).index(st.session_state.selected_token) if st.session_state.selected_token else 0)

        st.write("Selected token:", st.session_state.selected_token)  # Display the selected token
        
        if st.session_state.selected_token:
            next_token_probs = st.session_state.language_model.model.get((st.session_state.selected_token,), {})
            adjusted_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(adjusted_probs.values())
            adjusted_probs = {k: v / total for k, v in adjusted_probs.items()}

            if next_token_probs:
                fig, ax = plt.subplots()
                tokens, probs = zip(*sorted(next_token_probs.items(), key=lambda x: x[1], reverse=True)[:10])
                adj_tokens, adj_probs = zip(*sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)[:10])
                bar_width = 0.35
                index = np.arange(len(tokens))
                
                # Plot original and adjusted probabilities side by side
                bar1 = ax.bar(index, probs, bar_width, label='Original Probabilities')
                bar2 = ax.bar(index + bar_width, adj_probs, bar_width, label='Adjusted Probabilities (Temperature)')

                plt.xticks(index + bar_width / 2, tokens, rotation=45, ha='right')
                plt.xlabel("Next Tokens")
                plt.ylabel("Probability")
                plt.legend()
                st.pyplot(plt)  # Display the probability bar chart
            else:
                st.write("No next token probabilities available for the selected token.")  # Handle case with no probabilities

        # Display learned n-grams with visualizations
        display_ngram_bar_chart(st.session_state.language_model.model)
        display_ngram_wordcloud(st.session_state.language_model.model)

# Run the main function to start the app
if __name__ == "__main__":
    main()

# Add footer at the bottom of the app
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
    unsafe_allow_html=True  # Allow HTML for styling the footer
)
