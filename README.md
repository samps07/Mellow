# 😶‍🌫️ Mellow: AI-Enhanced Anonymous Discussion Board

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([(https://mellow-app.streamlit.app/)])
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-green)
![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-orange)

**Mellow** is a privacy-focused discussion platform that leverages modern AI to enhance user interactions while maintaining complete anonymity. It features AI-powered grammar correction, semantic search, and automatic content moderation.

Mellow is like Reddit, but:
* no accounts thus Privacy first
* AI cleans up your text
* AI blocks abuse before posting
* search works by meaning, not keywords

🔗 **Live Demo:** [Click here to visit Mellow](https://mellow-app.streamlit.app/)

---

## ✨ Key Features

* **🔒 True Anonymity:** No login required. Users receive a cryptographic deletion token to manage their content.
* **🤖 Geminify (AI Editing):** Uses **Google Gemini 2.5 Flash** to auto-correct grammar and spelling without altering the user's tone.
* **🧠 Semantic Search (RAG):** vectorized search allows users to find posts by *meaning*, not just keywords. (Powered by `sentence-transformers` and Postgres Vector).
* **🛡️ AI Moderation:** Real-time safety checks using Gemini to prevent harassment and hate speech.
* **🏷️ Auto-Tagging & Sentiment:** uses Zero-Shot Classification and DistilBERT to automatically categorize posts and detect mood.
* **☁️ Cloud Native:** Deployed on Streamlit Community Cloud with a persistent Supabase (PostgreSQL) database.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Language:** Python
* **Database:** Supabase (PostgreSQL + pgvector)
* **AI/LLM:** Google Gemini 2.5 Flash
* **Embeddings:** `all-MiniLM-L6-v2` (Sentence Transformers)
* **NLP:** Hugging Face Transformers (Sentiment Analysis)

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/mellow.git](https://github.com/yourusername/mellow.git)
    cd mellow
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your keys:
    ```env
    GEMINI_API_KEY="your_google_api_key"
    GEMINI_MODEL="gemini-2.5-flash"
    DB_URL="postgresql://user:password@host:port/postgres"
    SECRET_KEY="random_string_for_hashing"
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

* `app.py`: Main Streamlit application entry point.
* `db.py`: Database connection manager (PostgreSQL).
* `rag.py`: RAG logic and Gemini integration.
* `embeddings.py`: Vector generation and semantic search logic.
* `tagging.py`: Keyword extraction and topic classification.
* `moderation.py`: AI-safety and profanity checks.

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.

## Screenshots

1.  **App UI:**
<img width="1900" height="1019" alt="Screenshot 2025-12-10 233902" src="https://github.com/user-attachments/assets/70f274f4-1d5b-4c7a-9c96-d3b4734ee2ba" />

2.  **Recent Posts:**
<img width="1893" height="982" alt="Screenshot 2025-12-10 233927" src="https://github.com/user-attachments/assets/e2698469-9f1e-41d4-a920-aedce3c57925" />


3.  **Post Creation Interface:**
<img width="1915" height="925" alt="Screenshot 2025-12-10 233958" src="https://github.com/user-attachments/assets/78f96e35-9376-48b1-9b22-41329deda772" />

4.  **Post Deletion Interface:**
<img width="1892" height="888" alt="Screenshot 2025-12-10 234010" src="https://github.com/user-attachments/assets/cf7c44f5-24df-4851-8547-dc7cfc9f6674" />

5.  **RAG based search results, Responses powered by LLM:**
<img width="1882" height="1017" alt="Screenshot 2025-12-10 234114" src="https://github.com/user-attachments/assets/5eaa9f84-04fe-4617-8239-81b4ad6dfdbe" />






---
*Created by [Samith N]*
