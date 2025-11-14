# 🎬 CineBot: Your Multi-Tool Movie Expert Agent 🍿  
**Capstone Project – Module 3: AI Engineering**  
*Purwadhika Digital Technology School*

CineBot is your **24/7 movie buddy**—a conversational AI that answers ANY film question with wit, facts, and perfect recommendations. Built with Python, Streamlit, and LangChain, it’s live right now:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cinebot-movie-expert-agent.streamlit.app/)

---
## ⏭️ Quick Demo

![2025-11-1423-16-11-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a6093121-6289-4ef8-bc16-122e22e127fe)



---
## ✨ Superpowers
- **Smart Multi-Tool Brain**  
  CineBot picks the **perfect tool** on the fly:
  - **“I’m in the mood for mind-bending sci-fi”** → **RAG Tool** dives into Qdrant vector DB for semantic matches.  
  - **“What movie made the most money in 1994?”** → **SQL Sub-Agent** writes & runs perfect SQLite queries.

- **Personality**  
  Friendly, enthusiastic, and a little sarcastic—like your coolest cinephile friend.

- **Instant Movie Posters**  
  Every recommendation arrives in a gorgeous Markdown table with **clickable posters**.

- **Smart Follow-Ups**  
  Spots patterns (“You LOVE Nolan!”) and asks clever next questions.

- **Full Transparency**  
  - Every click traced in **Langfuse Cloud**.  
  - Expanders in the UI reveal **exact tool chosen**, **SQL generated**, and **why**.

- **Slick Streamlit UI**  
  Chat history, example prompts, and a helpful sidebar.

---
## 🏗️ How It Works (5-Second Architecture)
```
You type → CineBot Agent → Picks Tool → Gets Raw Data → Formats Answer → Streams to You
   ↑            ↓
Qdrant Cloud   ←→   SQLite DB
```
- **RAG Tool**: `get_movie_recommendations` → vector similarity → top-5 gems.  
- **SQL Tool**: `get_factual_movie_data` → LangChain SQL agent → natural-language answer + raw query.

---
## 🛠️ Tech Stack
- **Python 3**  
- **Streamlit** – instant web UI  
- **LangChain** – agent orchestration  
- **OpenAI** – `gpt-4o-mini` + `text-embedding-3-small`  
- **Qdrant Cloud** – vector search  
- **SQLite** – 1,000-movie fact vault  
- **Langfuse** – observability  
- **Pandas + SQLAlchemy**

---
## 🚀 Run Locally in 2 Minutes
```bash
# 1. Clone
git clone https://github.com/thariqabe666/cinebot-capstone.git
cd cinebot-capstone

# 2. Virtual env
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt
```

Create `.env` (root folder):
```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://your-qdrant-cluster.example.com
QDRANT_API_KEY=your_qdrant_key
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

```bash
# 4. One-time DB setup
python setup.py   # ← cleans, builds SQLite, fills Qdrant

# 5. LAUNCH!
streamlit run main.py
```

Browser opens → start chatting!

---
## ☁️ Deploy to Streamlit Cloud (Free)
1. Push everything to a **public** GitHub repo.  
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App → link repo.  
3. **Secrets** tab → paste the same keys (no quotes).  
4. Hit **Deploy** → share your CineBot with the world!

---
## 👨‍💻 Author
**Thariq Ahmad Baihaqi Adinegara**  
AI Engineering Student – Purwadhika Digital Technology School

---
## 🙏 Credits
- Dataset: [IMDb Top 1000 Movies](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) by Harshit Shankhdhar  
- Amazing libraries: Streamlit, LangChain, OpenAI, Qdrant, Langfuse  
- Big thanks to **Purwadhika** instructors!

Lights, camera, **chat**! 🎥  
Ask CineBot anything—“Funniest 80s comedy with zero explosions?”—and watch the magic.


