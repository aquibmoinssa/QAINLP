import streamlit as st
from preprocessing import extract_text_from_pdf, split_sentences, load_model, get_embeddings

st.set_page_config(page_title="Quantum NLP Explorer", layout="wide")
st.title("🔬 Stage 1: Classical NLP Preprocessing")

uploaded_file = st.file_uploader("Upload PDF:", type=["pdf"])
manual_text = st.text_area("Or paste text manually:")

if uploaded_file or manual_text:
    text = extract_text_from_pdf(uploaded_file) if uploaded_file else manual_text
    st.write(text[:1000])

    sentences = split_sentences(text)
    st.success(f"Extracted {len(sentences)} sentences")

    if st.checkbox("Show first 10 sentences"):
        for s in sentences[:10]:
            st.markdown(f"- {s}")

    if st.button("Generate Embeddings"):
        tokenizer, model = load_model()
        embeddings = get_embeddings(sentences, tokenizer, model)
        st.success(f"Generated {len(embeddings)} embeddings")