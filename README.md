# 🎬 CineBot: Agen Pakar Film Multi-Tool 🍿

**Sebuah Capstone Project untuk Modul 3 Program AI Engineering di Purwadhika Digital Technology School.**

CineBot adalah agen AI percakapan yang dirancang untuk menjadi teman diskusi film Anda. Dibangun menggunakan Python, Streamlit, dan LangChain, CineBot mampu menjawab berbagai pertanyaan seputar film, mulai dari rekomendasi berdasarkan *mood* atau kemiripan hingga data faktual yang spesifik.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cinebot-movie-expert-agent.streamlit.app/) 

---

## ✨ Fitur Utama

* **Agen Multi-Tool Cerdas:** CineBot menggunakan *agent* LangChain yang mampu memilih *tool* paling tepat secara dinamis:
    * **Rekomendasi Semantik (RAG):** Menggunakan Qdrant Vector Database untuk menemukan film berdasarkan deskripsi, tema, atau kemiripan plot.
    * **Analisis Data Faktual (Text-to-SQL):** Menggunakan *sub-agent* SQL untuk menjawab pertanyaan spesifik tentang rating, tahun rilis, pendapatan, sutradara, dll., dari database SQLite.
* **Persona Menarik:** CineBot dirancang dengan *prompt* khusus untuk memiliki kepribadian yang ramah, antusias, dan *witty* seperti teman penggemar film.
* **Tampilan Visual Poster:** Jawaban rekomendasi dan daftar film disajikan dalam tabel Markdown yang rapi, lengkap dengan gambar poster film.
* **Follow-up Proaktif:** CineBot menganalisis riwayat percakapan untuk mengidentifikasi pola (misalnya, preferensi sutradara) dan mengajukan pertanyaan lanjutan yang relevan, bukan sekadar "Ada lagi?".
* **Observability & Transparansi:**
    * **Langfuse Integration:** Setiap interaksi dilacak secara otomatis ke Langfuse Cloud, memungkinkan *tracing* dan *debugging* mendalam.
    * **Proses Berpikir:** Fitur *expander* di UI Streamlit menunjukkan *tool* mana yang dipilih agent, inputnya, dan (jika relevan) *query SQL* yang di-generate.
* **UI Interaktif:** Dibangun dengan Streamlit, menampilkan riwayat chat, tombol contoh pertanyaan, dan *sidebar* informatif.

---

## 🏗️ Arsitektur

Aplikasi ini menggunakan arsitektur agent multi-tool berbasis LangChain (menggunakan `create_agent`):

1.  **Input Pengguna:** Diterima melalui UI Streamlit.
2.  **Agent Utama (CineBot):** Menganalisis input dan riwayat, lalu berdasarkan `SYSTEM_PROMPT` yang canggih, memutuskan *tool* mana yang paling sesuai (RAG atau SQL).
3.  **Tool Routing:**
    * **Jika Kualitatif:** Memanggil `get_movie_recommendations` (RAG Tool) yang melakukan *similarity search* ke **Qdrant Cloud**.
    * **Jika Faktual:** Memanggil `get_factual_movie_data` (SQL Tool). Tool ini berisi **sub-agent SQL** yang:
        * Menggunakan `SQLDatabaseToolkit`.
        * Berinteraksi dengan database **SQLite** (`movies.db`).
        * Men-generate dan mengeksekusi *query SQL*.
        * Mengembalikan jawaban natural language + *query SQL*-nya.
4.  **Sintesis Jawaban:** Agent utama menerima output mentah dari *tool* (termasuk tag `||POSTER||` dan `||SQL_QUERY||`), lalu menyusun jawaban akhir sesuai format yang diperintahkan (tabel Markdown dengan sintaks gambar poster).
5.  **Output:** Ditampilkan di UI Streamlit, dengan *tracing* dikirim ke Langfuse di latar belakang.

---

## 🛠️ Tech Stack

* **Bahasa:** Python 3.x
* **Framework Aplikasi Web:** Streamlit
* **Framework AI/LLM:** LangChain
* **Model LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
* **Database Vektor:** Qdrant Cloud
* **Database Relasional:** SQLite
* **Observability/Tracing:** Langfuse Cloud
* **Manajemen Environment:** `venv`
* **Lainnya:** Pandas, SQLAlchemy

---

## 🚀 Setup & Instalasi

1.  **Clone Repositori:**
    ```bash
    git clone [https://github.com/NAMA_USER_GITHUB_KAMU/NAMA_REPO_KAMU.git](https://github.com/NAMA_USER_GITHUB_KAMU/NAMA_REPO_KAMU.git)
    cd NAMA_REPO_KAMU
    ```
2.  **Buat & Aktifkan Virtual Environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Catatan: Pastikan kamu sudah membuat file `requirements.txt` dengan `pip freeze > requirements.txt`)*
4.  **Siapkan Environment Variables:**
    Buat file `.env` di *root* direktori proyek dan isi dengan API keys/URL Anda:
    ```.env
    OPENAI_API_KEY="sk-..."
    QDRANT_URL="httpsPOST_URL_DARI_QDRANT_CLOUD"
    QDRANT_API_KEY="API_KEY_DARI_QDRANT_CLOUD"
    LANGFUSE_PUBLIC_KEY="pk-lf-..."
    LANGFUSE_SECRET_KEY="sk-lf-..."
    # LANGFUSE_HOST="[https://cloud.langfuse.com](https://cloud.langfuse.com)" # (Opsional, sesuaikan jika perlu)
    ```
5.  **Inisialisasi Database (Hanya Sekali):**
    Jalankan script `setup.py` untuk membersihkan data, membuat database SQLite (`movies.db`), dan mengisi koleksi di Qdrant Cloud.
    ```bash
    python setup.py
    ```

---

## ▶️ Menjalankan Aplikasi

Setelah setup selesai, jalankan aplikasi Streamlit:
```bash
streamlit run main.py
````

Aplikasi akan terbuka di browser lokal Anda.

-----

## ☁️ Deployment

Aplikasi ini dirancang untuk di-deploy ke **Streamlit Community Cloud**.

1.  Pastikan semua kode sudah di-*push* ke repositori GitHub **publik**.
2.  Pastikan file `requirements.txt` sudah benar dan ter-push.
3.  Di [Streamlit Cloud](https://share.streamlit.io/), buat aplikasi baru dan hubungkan ke repositori GitHub Anda.
4.  **PENTING:** Konfigurasikan **Secrets** di pengaturan aplikasi Streamlit Cloud Anda. Masukkan *key* dan *value* yang sama seperti di file `.env` (tanpa tanda kutip):
      * `OPENAI_API_KEY`
      * `QDRANT_URL`
      * `QDRANT_API_KEY`
      * `LANGFUSE_PUBLIC_KEY`
      * `LANGFUSE_SECRET_KEY`
      * `LANGFUSE_HOST` (Jika menggunakan host non-default)

-----

## 👨‍💻 Author

  * **Thariq Ahmad Baihaqi Adinegara**
  * *AI Engineering Student - Purwadhika Digital Technology School*

-----

## 🙏 Acknowledgements

  * Dataset: [IMDb Dataset of Top 1000 Movies and TV Shows](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows) oleh Harshit Shankhdhar di Kaggle.
  * Frameworks & Libraries: Streamlit, LangChain, OpenAI, Qdrant, Langfuse.
  * Purwadhika Digital Technology School.

