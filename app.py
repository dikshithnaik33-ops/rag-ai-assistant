import streamlit as st
from main import ToyotaRAGApp

st.title("🚗 Toyota AI Assistant")

if "rag_app" not in st.session_state:
    app = ToyotaRAGApp()
    app.setup(force_rebuild=False)
    st.session_state.rag_app = app

query = st.text_input("Ask your question:", key="input1")

if st.button("Ask"):
    if query:
        st.write("⏳ Processing...")
        response = st.session_state.rag_app.ask(query)
        st.write("### Answer:")
        st.write(response)