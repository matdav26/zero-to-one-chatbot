import streamlit as st
from PIL import Image
from Arthur_AI_chatbot import answer_query, index  # ✅ Import from your backend file

# Set Streamlit page config
st.set_page_config(page_title="Zero to One RAG Bot", page_icon="🎙️")

# Load and place logo + title using columns
col1, col2 = st.columns([1, 6])

with col1:
    logo = Image.open("Podcast_Logo.jpg")  # Update path if needed
    st.image(logo, width=150)  # Smaller width for alignment

with col2:
    st.markdown("## 🎙️ Zero to One Podcast Chatbot")  # Markdown title next to logo

# Input field for query
query = st.text_input("Ask a question:")

# Handle query
if query:
    st.write("Thinking...")
    try:
        answer, sources = answer_query(query, index)  # ✅ Unpack both values
        st.success(answer)

        with st.expander("🔍 Show source excerpts"):
            st.markdown(sources)

        st.markdown("*✅ This answer was validated using ArthurAI's hallucination detection.*")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Suggested questions below input
st.markdown("#### 💡 Example questions:")
st.markdown("""
- *What does Ilan Abehassera say about timing when starting a company?*  
- *How to find product market fit?*  
- *What are the differences between launching in the US vs. France?*  
""")
