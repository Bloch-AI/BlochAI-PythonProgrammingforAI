import streamlit as st
import time
import random
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

class SimpleLanguageModel:
    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(int))
        self.vocab = set()

    def train(self, text):
        tokens = word_tokenize(text.lower())
        self.vocab.update(tokens)
        for i in range(len(tokens) - 1):
            self.model[tokens[i]][tokens[i + 1]] += 1

    def generate(self, start_token, length=10, temperature=1.0):
        current_token = start_token
        result = [current_token]
        for _ in range(length - 1):
            next_token_probs = self.model[current_token]
            if not next_token_probs:
                break
            next_token_probs = {k: v ** (1 / temperature) for k, v in next_token_probs.items()}
            total = sum(next_token_probs.values())
            next_token_probs = {k: v / total for k, v in next_token_probs.items()}
            next_token = random.choices(list(next_token_probs.keys()), 
                                        weights=list(next_token_probs.values()))[0]
            result.append(next_token)
            current_token = next_token
        return result

def simulate_attention(tokens, output_tokens):
    attention_matrix = [[random.random() for _ in range(len(tokens))] for _ in range(len(output_tokens))]
    for row in attention_matrix:
        total = sum(row)
        for i in range(len(row)):
            row[i] /= total
    return attention_matrix

def main():
    st.title("LLM Demo App")

    # Initialize or load the language model
    if 'language_model' not in st.session_state:
        st.session_state.language_model = SimpleLanguageModel()

    # Input text
    input_text = st.text_area("Enter your text here:", height=100)

    # Model parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    with col2:
        output_length = st.slider("Output Length", 5, 50, 20)

    if st.button("Process"):
        if input_text:
            # Tokenization
            tokens = word_tokenize(input_text.lower())
            st.subheader("Tokenization")
            st.write("Tokens:", tokens)

            # Train the language model
            st.session_state.language_model.train(input_text)
            st.write("Model:", dict(st.session_state.language_model.model))

            # Generate output
            st.subheader("LLM Output")
            output_area = st.empty()
            progress_bar = st.progress(0)

            if tokens:
                output_tokens = st.session_state.language_model.generate(tokens[0], output_length, temperature)
                st.write("Output Tokens:", output_tokens)
                for i, token in enumerate(output_tokens):
                    output_area.write(" ".join(output_tokens[:i+1]))
                    progress_bar.progress((i + 1) / len(output_tokens))
                    time.sleep(0.2)  # Simulate processing time

                st.success("Processing complete!")

                # Attention visualization
                st.subheader("Attention Visualization")
                attention_matrix = simulate_attention(tokens, output_tokens)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=output_tokens, ax=ax, cmap="YlOrRd")
                plt.xlabel("Input Tokens")
                plt.ylabel("Output Tokens")
                st.pyplot(fig)

                # Token probability visualization
                st.subheader("Token Probability Visualization")
                start_token = st.selectbox("Select a token to see next token probabilities:", list(st.session_state.language_model.vocab))
                next_token_probs = st.session_state.language_model.model[start_token]
                if next_token_probs:
                    fig, ax = plt.subplots()
                    tokens, probs = zip(*sorted(next_token_probs.items(), key=lambda x: x[1], reverse=True)[:10])
                    ax.bar(tokens, probs)
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel("Next Tokens")
                    plt.ylabel("Probability")
                    st.pyplot(fig)
                else:
                    st.write("No next token probabilities available for the selected token.")
            else:
                st.warning("Tokenization produced no tokens. Please enter valid text.")

        else:
            st.warning("Please enter some text to process.")

if __name__ == "__main__":
    main()

# Add footer
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
