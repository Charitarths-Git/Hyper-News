import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="HyperNews AI", page_icon="🧠", layout="wide")

# Custom CSS for glassmorphism and animations
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(5px);
    -webkit-backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-5px);
}
.category-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
    background: rgba(88, 101, 242, 0.2);
    color: #aeb5ff;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 HyperNews — Hyper-Personalized News")

# Sidebar
st.sidebar.header("🎛️ Demo Controls")
user_id = st.sidebar.text_input("User ID", value="demo_user_1")
mood = st.sidebar.selectbox("😊 Simulate Mood", ["neutral", "curious", "happy", "stressed", "tired"])
query = st.sidebar.text_input("🔍 Search Topic (RAG)", placeholder="Optional topic...")

def get_time_of_day() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:   return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else:                 return "night"

st.sidebar.markdown(f"🕐 Detected Time: **{get_time_of_day().capitalize()}**")

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'explanation' not in st.session_state:
    st.session_state.explanation = None

if st.sidebar.button("🚀 Regenerate Feed", use_container_width=True):
    with st.spinner("Analyzing Knowledge Graph & Retrieving Context..."):
        try:
            resp = requests.post("http://localhost:8000/recommend", json={
                "user_id": user_id, 
                "mood": mood, 
                "query": query if query else None, 
                "n": 6
            })
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.recommendations = data.get("articles", [])
                st.session_state.explanation = data.get("explanation", "")
            else:
                st.error("Backend API Error")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Is FastAPI running on port 8000?")

if st.session_state.explanation:
    st.markdown(f"<div style='padding: 15px; border-left: 5px solid #aeb5ff; background: rgba(88, 101, 242, 0.1); border-radius: 8px; margin-bottom: 25px;'>💡 <b>AI Reasoning:</b> {st.session_state.explanation}</div>", unsafe_allow_html=True)

if st.session_state.recommendations:
    cols = st.columns(2)
    for i, article in enumerate(st.session_state.recommendations):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="glass-card">
                <div class="category-badge">{article['category']}</div>
                <h3>{article['title']}</h3>
                <p style="color: #ccc; font-size: 14px;">{article.get('abstract', '')[:100]}...</p>
                <div style="font-size: 11px; color: #888; margin-top: 10px;">Relevance Score: {article.get('score', 0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            if c1.button("👍 Read", key=f"read_{article['news_id']}"):
                try:
                    requests.post("http://localhost:8000/feedback", json={
                        "user_id": user_id, "article_id": article['news_id'], "action": "read_full"
                    })
                    st.toast("Feedback recorded!")
                except: pass
            if c2.button("⏩ Skip", key=f"skip_{article['news_id']}"):
                 try:
                    requests.post("http://localhost:8000/feedback", json={
                        "user_id": user_id, "article_id": article['news_id'], "action": "skip"
                    })
                    st.toast("Skipped recorded!")
                 except: pass
            if c3.button("⭐ Save", key=f"save_{article['news_id']}"):
                 try:
                    requests.post("http://localhost:8000/feedback", json={
                        "user_id": user_id, "article_id": article['news_id'], "action": "save"
                    })
                    st.toast("Saved!")
                 except: pass
elif st.session_state.recommendations == []:
    st.info("No articles found matching your criteria.")
else:
     st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h2>Welcome to HyperNews</h2>
            <p>Click 'Regenerate Feed' in the sidebar to get your personalized news.</p>
        </div>
     """, unsafe_allow_html=True)
