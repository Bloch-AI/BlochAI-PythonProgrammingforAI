def main():
    st.title("💻 LLM Simulator App")

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
            # Tokenization and training
            tokens, embeddings = st.session_state.language_model.train(input_text)
            
            # Store the tokens and generated data in session state to avoid recomputation
            st.session_state.tokens = tokens
            st.session_state.embeddings = embeddings
            start_tokens = tokens[:2]  # Starting with the first two tokens
            output_with_rationales = st.session_state.language_model.generate(start_tokens, output_length, temperature)
            st.session_state.output_with_rationales = output_with_rationales
            st.session_state.generated_tokens = [entry["token"] for entry in output_with_rationales]
            st.session_state.generated_sentence = " ".join(st.session_state.generated_tokens)

            # Debugging: Print the model contents
            print("Model contents:", st.session_state.language_model.model)
        
    if 'output_with_rationales' in st.session_state:
        st.subheader("LLM Output")
        st.write("Generated Output Sentence:", st.session_state.generated_sentence)

        with st.expander("Output Tokens with Rationales", expanded=True):
            for i, entry in enumerate(st.session_state.output_with_rationales):
                st.write(f"{i}: {entry['token']} - {entry['rationale']}")

        st.subheader("Attention Visualization")
        st.write("The attention visualization shows how each output token attends to each input token.")
        attention_matrix = simulate_attention(st.session_state.tokens, st.session_state.generated_tokens)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_matrix, xticklabels=st.session_state.tokens, yticklabels=st.session_state.generated_tokens, ax=ax, cmap="YlOrRd")
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        st.pyplot(fig)

        st.subheader("Token Probability Visualization")
        st.write("The token probability visualization shows the likelihood of each possible next token given the current token, based on the training data.")
        
        if 'selected_token' not in st.session_state:
            st.session_state.selected_token = None
        
        st.session_state.selected_token = st.selectbox("Select a token to see next token probabilities:", list(st.session_state.language_model.vocab), index=list(st.session_state.language_model.vocab).index(st.session_state.selected_token) if st.session_state.selected_token else 0)

        # Debugging: Print the selected token and check if it exists in the model
        print("Selected token:", st.session_state.selected_token)
        
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
                
                bar1 = ax.bar(index, probs, bar_width, label='Original Probabilities')
                bar2 = ax.bar(index + bar_width, adj_probs, bar_width, label='Adjusted Probabilities (Temperature)')

                plt.xticks(index + bar_width / 2, tokens, rotation=45, ha='right')
                plt.xlabel("Next Tokens")
                plt.ylabel("Probability")
                plt.legend()
                st.pyplot(fig)
            else:
                st.write("No next token probabilities available for the selected token.")

if __name__ == "__main__":
    main()
