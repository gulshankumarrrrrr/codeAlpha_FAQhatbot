import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 Custom dark styling
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #f8f8f2;
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }

    .stTextInput>div>div>input {
        background-color: #1e1e1e;
        color: #00ffe7;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px;
    }

    .stButton>button {
        background-color: #00bfff;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 10px 25px;
        border: none;
        font-size: 16px;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #009acb;
        transform: scale(1.05);
    }

    .stMarkdown h1 {
        color: #00f7ff;
        text-shadow: 0 0 10px #00f7ff;
        text-align: center;
    }

    .stMarkdown h4 {
        color: #b4f8c8;
    }
    </style>
""", unsafe_allow_html=True)

# 🤖 Load data
with open("data.json", "r") as file:
    data = json.load(file)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# 🔍 TF-IDF + Similarity
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# 🧠 Chatbot UI
st.markdown("<h1>🤖 FAQ Chatbot</h1>", unsafe_allow_html=True)
st.markdown("### 💬 Ask me anything from my knowledge base:")

user_input = st.text_input("🧠 Your Question:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(input_vec, X)
    index = similarity.argmax()

    if similarity[0][index] > 0.2:
        st.success(f"✅ {answers[index]}")
    else:
        st.error("❌ Sorry! I couldn't find the answer.")
