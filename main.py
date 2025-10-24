import streamlit as st
import os
from dotenv import load_dotenv
import uuid  # <-- TAMBAHAN LANGFUSE (untuk session_id unik)

# --- Import untuk LLM, Tools, dan Agent ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.utilities import SQLDatabase
# === IMPORT BARU DARI DOCS ===
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage  # <-- SUDAH ADA (penting)
# <-- TAMBAHAN LANGFUSE ---
from langfuse import get_client
from langfuse.langchain import CallbackHandler
# -------------------------


# === BAGIAN 1: SETUP & INISIALISASI ===
# Muat environment variables
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
# (Keys Langfuse akan dibaca otomatis oleh get_client() dari secrets)
    print("Loaded keys from Streamlit secrets.")
except KeyError:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# (Keys Langfuse akan dibaca otomatis oleh get_client() dari .env)
    print("Loaded keys from .env file.")

# <-- MODIFIKASI LANGFUSE v3: Inisialisasi klien global ---
# Ini akan membaca LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, dll.
# dari environment (secrets/dotenv) secara otomatis.
try:
    langfuse = get_client()
except Exception as e:
    print(f"Peringatan: Gagal menginisialisasi Langfuse. Tracing mungkin tidak aktif. Error: {e}")
    langfuse = None
# -----------------------------------------------------

# Inisialisasi model LLM dan Embedding
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# Tentukan konstanta
QDRANT_COLLECTION_NAME = "imdb_movies"
SQL_DB_URI = "sqlite:///movies.db"


# === BAGIAN 2: DEFINISI TOOLS ===

@tool
def get_movie_recommendations(question: str) -> str:
    """
    Gunakan alat ini untuk mencari rekomendasi film berdasarkan deskripsi plot, 
    tema, genre, atau film lain yang mirip. 
    Input harus berupa pertanyaan dalam bahasa natural tentang film yang dicari.
    Contoh: 'Cari film tentang perjalanan waktu' atau 'Rekomendasi film mirip The Dark Knight'.
    """
    print(f"\n>> MENGGUNAKAN TOOL RAG: {question}")
    qdrant_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    results = qdrant_store.similarity_search(question, k=3)
    formatted_results = "\n\n".join(
        [
            f"Judul: {doc.metadata.get('title', 'N/A')}\n"
            f"Tahun: {doc.metadata.get('year', 'N/A')}\n"
            f"Rating: {doc.metadata.get('rating', 'N/A')}\n"
            f"Genre: {doc.metadata.get('genre', 'N/A')}\n"
            f"Sinopsis: {doc.page_content.split('Sinopsis: ')[-1]}"
            f"||POSTER||{doc.metadata.get('poster', 'No Poster URL')}"
            for doc in results
        ]
    )
    return f"Berikut adalah 3 film yang paling relevan berdasarkan pencarianmu:\n{formatted_results}"

# === FUNGSI TOOL SQL YANG DIMODIFIKASI SESUAI DOCS BARU ===
@tool
def get_factual_movie_data(question: str) -> str:
    """
    Gunakan alat ini untuk menjawab pertanyaan spesifik dan faktual tentang data film, 
    seperti rating, tahun rilis, sutradara, pendapatan (gross), jumlah vote, dan durasi. 
    Sangat baik untuk pertanyaan yang melibatkan angka, statistik, perbandingan, atau daftar.
    Contoh: 'top 5 film rating tertinggi 2019', 'rata-rata pendapatan film Christopher Nolan', 'total film di atas 150 menit'.
    """
    print(f"\n>> MENGGUNAKAN TOOL SQL (Metode Toolkit Manual): {question}")
    
    db = SQLDatabase.from_uri(SQL_DB_URI)
    
    # 1. Buat toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 2. Dapatkan daftar tools-nya
    sql_tools = toolkit.get_tools()

    # 3. Buat system prompt khusus untuk SQL
    sql_system_prompt = """
    You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run,
    then look at the results of the query and return the answer. Unless the user
    specifies a specific number of examples they wish to obtain, always limit your
    query to at most {top_k} results.

    You can order the results by a relevant column to return the most interesting
    examples in the database. Never query for all the columns from a specific table,
    only ask for the relevant columns given the question.

    When you query for data about specific movies (e.g., Series_Title, Rating), 
    YOU MUST ALWAYS ALSO SELECT the 'Poster_Link' column.
    In your final natural language answer, after mentioning a movie, 
    YOU MUST include its poster URL, prefixed with the special tag '||POSTER||'.
    Contoh Jawaban: "Filmnya adalah The Dark Knight. ||POSTER||http://url.com/poster.jpg"

    You MUST double check your query before executing it. If you get an error while
    executing a query, rewrite the query and try again.
    
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
    database.
    
    To start you should ALWAYS look at the tables in the database to see what you
    can query. Do NOT skip this step.
    Then you should query the schema of the most relevant tables.
    """.format(
        dialect=db.dialect,
        top_k=5,
    )

    # 4. Buat "sub-agent" khusus SQL
    sql_agent_runnable = create_agent(
        llm,
        sql_tools,
        system_prompt=sql_system_prompt,
    )
    
    try:
        # 5. Jalankan sub-agent SQL
        response_state = sql_agent_runnable.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        
        # === TAMBAHAN BARU: EKSTRAK JAWABAN & QUERY ===
        
        # 6. Ekstrak jawaban AKHIR dari pesan terakhir
        answer = response_state["messages"][-1].content
        
        # 7. Ekstrak SQL Query dari intermediate steps (history pesan si sub-agent)
        sql_query = "Tidak ada query SQL yang dieksekusi (jawaban langsung)."
        # Iterasi mundur untuk menemukan tool call terakhir ke 'sql_db_query'
        # --- PERBAIKAN: Tambahkan `isinstance` untuk mencegah error ---
        for msg in reversed(response_state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    if call['name'] == 'sql_db_query':
                        sql_query = call['args'].get('query', 'Query tidak ditemukan')
                        break  # Menemukan query, hentikan loop 'for call'
                if "query tidak ditemukan" not in sql_query and "Tidak ada query" not in sql_query:
                    break # Hentikan iterasi
            
        # 8. Gabungkan dengan delimiter unik "||SQL_QUERY||"
        # Ini adalah "trik" agar kita bisa mengirim DUA informasi (jawaban dan query)
        # kembali ke agent utama sebagai SATU string.
        return f"{answer}||SQL_QUERY||{sql_query}"
        
    except Exception as e:
        # Kirim error dengan delimiter yang sama agar parsing tidak gagal
        return f"Terjadi error saat menjalankan query: {e}.||SQL_QUERY||"  


# Daftarkan semua tools yang dimiliki agent
tools = [get_movie_recommendations, get_factual_movie_data]


# === BAGIAN 3: MERAKIT AGENT UTAMA ===

# 1. Definisikan System Prompt untuk Agent Utama
SYSTEM_PROMPT = """
You are CineBot, an enthusiastic, witty, and super helpful movie expert.
Your personality is like a close friend who is a total movie buff. Use varied, friendly, engaging language, and casual slang (e.g., "wah", "keren", "epik", "bikin mikir", "wajib tonton", "nendang", "spill dong").
**JANGAN** terlalu sering mengulang kata yang sama (contoh: 'gokil'). Variasikan bahasamu!

--- **ATURAN TOOL WAJIB (SANGAT PENTING)** ---
You have two tools. You MUST choose ONLY ONE tool per user question. Perhatikan baik-baik perbedaannya:

1.  **get_movie_recommendations (Tool KUALITATIF)**:
    * Gunakan HANYA untuk pertanyaan kualitatif, berbasis *tema*, *plot*, atau *kemiripan*.
    * **Contoh:** 'Cari film *tentang* perjalanan waktu', 'Rekomendasi film *mirip* Inception', 'film yang *sebagus* Interstellar'.

2.  **get_factual_movie_data (Tool KUANTITATIF / FAKTUAL)**:
    * Gunakan untuk pertanyaan faktual, data spesifik, atau *daftar* berdasarkan *fakta*.
    * **INI JUGA TERMASUK** jika user meminta 'rekomendasi' atau 'daftar film' berdasarkan **fakta spesifik** seperti **Sutradara, Aktor, Tahun Rilis, atau Genre.**
    * **Contoh:** 'Siapa sutradara film The Dark Knight?', 'Top 5 film gross tertinggi?', **'Rekomendasi film Christopher Nolan'**, **'Kasih tau semua film Tom Hanks'**, **'Daftar film genre Sci-Fi terbaik'**.

--- **ATURAN FORMAT JAWABAN (PENTING!)** ---
When your answer comes from a tool (RAG or SQL):
1.  **Opener:** Mulai dengan sapaan yang antusias dan personal. (Contoh: "Wah, Christopher Nolan emang jagonya bikin film epik yang bikin otak muter! Siap, ini dia daftar film-filmnya...")
2.  **The List (WAJIB TABEL):** Kamu HARUS menyajikan daftar film dalam format **Tabel Markdown**.
    
    --- **INSTRUKSI POSTER (KRUSIAL!)** ---
    * Tool akan memberimu data mentah yang mengandung tag `||POSTER||http://...`
    * Tugasmu adalah **mengambil URL** dari tag tersebut dan mengubahnya menjadi **sintaks gambar Markdown** (`![Poster](URL)`) di dalam tabel.
    * Buat kolom baru bernama `Poster` untuk menaruh sintaks gambar itu.
    * Jika URL poster tidak ada atau `No Poster URL`, tulis "N/A" di kolom Poster.
    
    * **Contoh Tabel YANG HARUS DIIKUTI:**
        | Poster | Film | Tahun | Rating | Kenapa Wajib Tonton? |
        |---|---|---|---|---|
        | ![Poster](httpsMARVEL_POSTER_URL.jpg) | Avengers: Endgame | 2019 | 8.4 | Puncak epik dari saga Marvel yang emosional dan penuh aksi. |
        | ![Poster](INTERSTELLAR_POSTER_URL.jpg) | Interstellar | 2014 | 8.6 | Sci-fi epik tentang waktu dan cinta keluarga. Visualnya luar biasa. |

--- **!!! ATURAN KRUSIAL: FOLLOW-UP CERDAS (Tiru ini!) !!!** ---
* **JANGAN PERNAH** mengakhiri jawabanmu dengan pertanyaan generik dan membosankan...
* **SELALU** akhiri dengan **saran proaktif** atau pertanyaan terbuka yang spesifik.
    **Contoh Follow-up YANG BAIK:**
    "**Rekomendasi Mulai Dari Mana?**
    * Kalau kamu suka [tema/film sebelumnya], coba **[Film A]** atau **[Film B]** dulu.
    * Pengen aksi epik? Tonton **[Film C]**.
    * Suka sci-fi emosional? **[Film D]** wajib banget.
    Udah nonton yang mana aja? Atau pengen aku kasih saran urutan nonton biar maksimal? 😄"

--- **ATURAN HISTORY (PROAKTIF)** ---
* Selalu periksa riwayat obrolan. Jika kamu melihat pola (misal, user bertanya 'Interstellar' lalu 'Inception'), **kamu HARUS membahasnya!**
* **Contoh:** "Eh, aku baru sadar nih... kamu kayaknya ngefans berat sama filmnya Christopher Nolan ya? Dua film tadi kan karya dia. Mau aku buatin daftar lengkap film-film dia yang lain?"
"""

# 2. Buat Agent (Cara Baru yang Jauh Lebih Simpel)
agent_runnable = create_agent(
    llm,
    tools,
    system_prompt=SYSTEM_PROMPT
)


# === BAGIAN 4: STREAMLIT UI (VERSI BARU DENGAN FIX hasattr) ===
st.title("🎬 CineBot: Movie Expert Agent 🍿")
st.write("Tanyakan apapun padaku! (Rekomendasi film atau data faktual)")

# <-- TAMBAHAN LANGFUSE: Buat session_id unik per sesi Streamlit ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# -----------------------------------------------------------------

# --- TAMBAHAN: Tombol Contoh Pertanyaan ---
def set_user_input(question):
    """Callback function to set user input from a button."""
    st.session_state.user_input = question

st.write("Atau, coba salah satu contoh ini:")
cols = st.columns([1, 1, 1.2]) # Buat kolom dengan lebar berbeda
with cols[0]:
    st.button("Film mirip Inception", on_click=set_user_input, args=("Rekomendasi film yang mirip Inception",), use_container_width=True)
with cols[1]:
    st.button("Top 5 film terlaris", on_click=set_user_input, args=("Apa 5 film dengan pendapatan (gross) tertinggi?",), use_container_width=True)
with cols[2]:
    st.button("Rekomendasi film Nolan", on_click=set_user_input, args=("Kasih tau daftar film dari Christopher Nolan",), use_container_width=True)

# Ambil input dari chat box atau dari state yang di-set oleh tombol
chat_input = st.chat_input("Contoh: 'Film mirip Inception' atau 'Top 5 film 2010'")
user_input = chat_input or st.session_state.get("user_input", None)

# Hapus state setelah digunakan agar tidak ter-trigger lagi
if "user_input" in st.session_state and not chat_input:
    del st.session_state.user_input
# --- AKHIR TAMBAHAN ---

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- PERBAIKAN 1: Konversi history dari dict ke objek LangChain ---
    # Agent runnable mengharapkan list of BaseMessage (HumanMessage, AIMessage), bukan dict.
    from langchain_core.messages import HumanMessage, AIMessage
    langchain_messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]

    # Variabel untuk menyimpan info proses berpikir
    tool_call_info = None
    full_tool_output = ""
    sql_query_to_display = None
    display_answer = ""
    last_valid_state = None 

    with st.chat_message("assistant"):
        with st.spinner("CineBot sedang mencari jawaban..."):
            
            # <-- MODIFIKASI LANGFUSE v3: Inisialisasi dan Config ---
            # 1. Inisialisasi handler KOSONG (sesuai docs baru)
            langfuse_handler = CallbackHandler()
            
# <-- PERUBAHAN LANGFUSE v3: Buat Config dengan Metadata ---
            config = {
                "callbacks": [langfuse_handler],
                "run_name": f"Query: {user_input[:30]}...",
                "metadata": { # Lewatkan atribut di sini
                    "langfuse_session_id": st.session_state.session_id,
                    "langfuse_user_id": st.session_state.session_id, 
                    "langfuse_tags": ["CineBot-v1", "Capstone-Mod3"]
                }
            }
            # -----------------------------------------------
            
            # 1. Gunakan .stream() DENGAN config
            stream = agent_runnable.stream(
                {"messages": langchain_messages},
                stream_mode="values",
                config=config  # <-- TAMBAHAN LANGFUSE
            )
            
            for chunk in stream:
                if "messages" in chunk:
                    last_valid_state = chunk 
                    
                    last_message = chunk["messages"][-1]
                    
                    # === INI ADALAH PERBAIKANNYA ===
                    # 3. Tangkap momen agent MEMUTUSKAN memanggil tool (dengan aman)
                    #    Kita cek dulu apakah atribut 'tool_calls' ada
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        call = last_message.tool_calls[0]
                        tool_call_info = {
                            "name": call['name'],
                            "args": call['args']
                        }
                    # === AKHIR PERBAIKAN ===
                    
                    # --- PERBAIKAN 2: Gunakan .type bukan .role ---
                    # Objek pesan LangChain memiliki atribut .type ('human', 'ai', 'tool')
                    if hasattr(last_message, "type") and last_message.type == "tool":
                        full_tool_output = last_message.content
            
            # 5. Ambil jawaban akhir (setelah stream selesai)
            if last_valid_state:
                final_answer_object = last_valid_state["messages"][-1]
                display_answer = final_answer_object.content
            else:
                display_answer = "Maaf, terjadi kesalahan."

            # 6. Logika Parsing untuk SQL
            if tool_call_info and tool_call_info['name'] == 'get_factual_movie_data':
                if "||SQL_QUERY||" in full_tool_output:
                    parts = full_tool_output.split("||SQL_QUERY||")
                    sql_query_to_display = parts[1]
                else:
                    sql_query_to_display = "Query tidak dapat diekstrak dari tool."

            # Tampilkan jawaban bersih di chat
            # --- MODIFIKASI DIMULAI DARI SINI ---
            
            # Cek apakah jawaban mengandung tag poster
            if "||POSTER||" in display_answer:
                # Pisahkan jawaban berdasarkan tag
                parts = display_answer.split("||POSTER||")
                
                # Tampilkan bagian pertama (teks sebelum poster pertama)
                st.markdown(parts[0])
                
                # Loop untuk sisa bagian (yang berisi URL poster dan teks setelahnya)
                for part in parts[1:]:
                    # Pisahkan URL dari sisa teks
                    # (Asumsi: URL adalah hal pertama setelah tag)
                    try:
                        # Split di baris baru pertama
                        url_part, *text_part = part.split('\n', 1) 
                        poster_url = url_part.strip()
                        
                        # Tampilkan gambar jika URL-nya valid
                        if poster_url.startswith("http"):
                            st.image(poster_url)
                        else:
                            st.markdown(f"*(Poster tidak tersedia: {poster_url})*")
                        
                        # Tampilkan sisa teks setelah poster
                        if text_part:
                            st.markdown(text_part[0])
                            
                    except ValueError:
                        # Jika ada bagian yang aneh, tampilkan saja sebagai teks
                        st.markdown(part)
            else:
                # Jika tidak ada tag poster, tampilkan seperti biasa
                st.markdown(display_answer)
            # --- MODIFIKASI SELESAI ---


    # --- Tampilkan Expander DI LUAR `chat_message` ---
    if tool_call_info:
        with st.expander("Lihat Proses Berpikir CineBot 🤖"):
            st.markdown(f"**Tool Dipilih:** `{tool_call_info['name']}`")
            st.markdown(f"**Input untuk Tool:**")
            st.json(tool_call_info['args'])
            
            if sql_query_to_display:
                st.markdown("**Generated SQL Query:**")
                st.code(sql_query_to_display.strip(), language="sql")
            
            st.markdown("**Output Mentah dari Tool:**")
            st.text(full_tool_output.split("||SQL_QUERY||")[0])

    # Tambahkan jawaban bersih (yang sudah disintesis) ke history
    st.session_state.messages.append({"role": "assistant", "content": display_answer})