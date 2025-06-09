import streamlit as st
from preprocessing import extract_text_from_pdf, split_sentences, get_sentence_embeddings

st.set_page_config(page_title="Quantum NLP Explorer", layout="wide")

st.title("🔬 Stage 1: Classical NLP Preprocessing")

# Upload PDF or enter text
uploaded_file = st.file_uploader("Upload a scientific paper (PDF):", type=["pdf"])
manual_text = st.text_area("Or paste text directly:", "")

if uploaded_file or manual_text:
    with st.spinner("Processing text..."):
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = manual_text

        st.subheader("Extracted Text Sample")
        st.write(text[:1000])  # preview

        sentences = split_sentences(text)
        st.success(f"{len(sentences)} sentences extracted.")

        if st.checkbox("Show sentences"):
            for s in sentences[:10]:
                st.markdown(f"• {s}")

        if st.button("Generate Embeddings"):
            embeddings = get_sentence_embeddings(sentences)
            st.success(f"Generated {len(embeddings)} sentence embeddings.")