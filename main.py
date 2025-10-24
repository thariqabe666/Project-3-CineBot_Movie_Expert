# === BAGIAN 1: SETUP & INISIALISASI ===
# 1.1: Imports & Requirements
# - Semua import yang diperlukan untuk Streamlit, LangChain, Qdrant, Langfuse, dotenv, dll.

import streamlit as st
import os
# from dotenv import load_dotenv
import uuid

# LangChain imports for LLM, Tools, and Agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

# Langfuse imports for tracing
from langfuse import get_client
from langfuse.langchain import CallbackHandler

# 1.2: Streamlit page configuration
# - Atur judul, ikon, dan layout halaman
st.set_page_config(
    page_title="CineBot 🎬",
    page_icon="🍿",
    layout="wide" # Mengubah layout menjadi 'wide'
)

# 1.3: Environment variables loading (secrets atau .env)
# - Prioritas: Streamlit secrets -> .env
# - Variabel yang digunakan: OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
# - Tujuan: memudahkan deployment (Streamlit Cloud) dan lokal (.env)
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    # Langfuse keys are automatically read by get_client() from secrets
    print("Environment variables loaded from Streamlit secrets.")
except KeyError:
    # load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    # Langfuse keys are automatically read by get_client() from .env
    print("Environment variables loaded from .env file.")

# 1.4: Langfuse client initialization (opsional)
# - Inisialisasi tracing client jika tersedia.
# - Jika gagal, tampilkan peringatan tetapi jalankan aplikasi tanpa tracing.
# Initialize Langfuse client globally for tracing
# Ini akan membaca LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, dll.
# dari environment (secrets/dotenv) secara otomatis.
try:
    langfuse = get_client()
except Exception as e:
    print(f"Peringatan: Gagal menginisialisasi Langfuse. Tracing mungkin tidak aktif. Error: {e}")
    langfuse = None

# 1.5: LLM & Embedding initialization
# - Konfigurasi model LLM (ChatOpenAI) dan embeddings (OpenAIEmbeddings).
# - Gunakan API key dari environment.
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

# 1.6: Konstanta aplikasi
# - Nama koleksi Qdrant dan URI database SQLite disimpan di sini.
# Define constants for Qdrant and SQL database
QDRANT_COLLECTION_NAME = "imdb_movies"
SQL_DB_URI = "sqlite:///movies.db"

# === BAGIAN 2: DEFINISI TOOLS ===
# 2.1: Overview
# - Tool adalah fungsi yang dipakai agent untuk mengambil data.
# - Di aplikasi ini ada dua tool: RAG (Qdrant) untuk rekomendasi kualitatif dan SQL untuk data faktual.

# 2.2: Tool RAG — get_movie_recommendations
# - Tujuan: cari film berdasarkan tema/plot/kemiripan (kualitatif).
# - Input: pertanyaan natural language.
# - Output: string terformat dengan metadata film, termasuk tag khusus poster `||POSTER||URL`.
@tool
def get_movie_recommendations(question: str) -> str:
    """
    Gunakan alat ini untuk mencari rekomendasi film berdasarkan deskripsi plot, 
    tema, genre, atau film lain yang mirip. 
    Input harus berupa pertanyaan dalam bahasa natural tentang film yang dicari.
    Contoh: 'Cari film tentang perjalanan waktu' atau 'Rekomendasi film mirip The Dark Knight'.
    """
    print(f"\n>> Using RAG Tool for movie recommendations: '{question}'")
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

# 2.3: Tool SQL — get_factual_movie_data
# - Tujuan: jawab pertanyaan faktual/kuantitatif (rating, tahun, sutradara, dsb.)
# - Pendekatan:
#   1) Buat koneksi SQLDatabase
#   2) Inisialisasi SQLDatabaseToolkit dan ambil tools SQL
#   3) Definisikan system prompt khusus SQL (guidelines untuk pembuatan query, pembatasan, dan instruksi Poster)
#   4) Buat sub-agent khusus untuk menjalankan langkah pembuatan query dan eksekusi
#   5) Jalankan sub-agent, ambil jawaban akhir dan, jika ada, query SQL yang dieksekusi
# - Output: gabungan jawaban dan delimiter `||SQL_QUERY||` diikuti SQL query (atau pesan error + delimiter).
@tool
def get_factual_movie_data(question: str) -> str:
    """
    Gunakan alat ini untuk menjawab pertanyaan spesifik dan faktual tentang data film, 
    seperti rating, tahun rilis, sutradara, pendapatan (gross), jumlah vote, dan durasi. 
    Sangat baik untuk pertanyaan yang melibatkan angka, statistik, perbandingan, atau daftar.
    Contoh: 'top 5 film rating tertinggi 2019', 'rata-rata pendapatan film Christopher Nolan', 'total film di atas 150 menit'.
    """ 
    print(f"\n>> Using SQL Tool for factual movie data: '{question}'")
    
    db = SQLDatabase.from_uri(SQL_DB_URI)
    
    # 1. Create SQL toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # 2. Get the tools from the toolkit
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

    # 4. Create a dedicated "sub-agent" for SQL queries
    sql_agent_runnable = create_agent(
        llm,
        sql_tools,
        system_prompt=sql_system_prompt,
    )
    
    try:
        # 5. Invoke the SQL sub-agent
        response_state = sql_agent_runnable.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        
        # 6. Extract the final answer from the last message
        answer = response_state["messages"][-1].content
        
        # 7. Extract the SQL Query from intermediate steps (sub-agent's message history)
        sql_query = "Tidak ada query SQL yang dieksekusi (jawaban langsung)."
        for msg in reversed(response_state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    if call['name'] == 'sql_db_query':
                        sql_query = call['args'].get('query', 'Query tidak ditemukan')
                        break # Found the query, stop inner loop
                if sql_query != "Tidak ada query SQL yang dieksekusi (jawasan langsung).": # If query was found, stop outer loop
                    break
            
        # 8. Combine answer and SQL query using a unique delimiter for main agent parsing
        return f"{answer}||SQL_QUERY||{sql_query}"
        
    except Exception as e:
        # Kirim error dengan delimiter yang sama agar parsing tidak gagal
        return f"Terjadi error saat menjalankan query: {e}.||SQL_QUERY||"  


# 2.4: Daftar tool yang diregistrasi ke agent utama
tools = [get_movie_recommendations, get_factual_movie_data]

# === BAGIAN 3: MERAKIT AGENT UTAMA ===
# 3.1: System prompt utama (PERSONALITAS + ATURAN PENTING)
# - Menetapkan persona CineBot (ramah, witty, informatif) dan aturan ketat penggunaan tool.
# - Penekanan pada:
#   * Pilih hanya SATU tool per pertanyaan
#   * Perbedaan kapan pakai RAG vs SQL
#   * Format jawaban: wajib tabel Markdown, kolom Poster harus berisi sintaks gambar Markdown atau "N/A"
#   * Instruksi follow-up cerdas dan aturan history
# - System prompt ini sangat menentukan perilaku agent.
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

# 3.2: Buat agent utama (runnable)
# - Gunakan create_agent dengan llm, tools, dan system_prompt di atas.
# - Hasil: agent_runnable yang dapat dipanggil / di-stream.
agent_runnable = create_agent(
    llm,
    tools,
    system_prompt=SYSTEM_PROMPT
)

# === BAGIAN 4: STREAMLIT UI & FLOW INTERAKSI ===
# 4.1: UI Sidebar
# - Informasi aplikasi, pembuat, link GitHub, tombol untuk menghapus riwayat obrolan.
# - Catatan: ketika hapus riwayat, set st.session_state.messages = [] dan rerun.
with st.sidebar:
    st.title("🎬 CineBot")
    st.info("Saya adalah CineBot, agen AI pakar film yang siap membantu Anda!")
    
    st.markdown("---")
    st.markdown("### Dibuat oleh:")
    st.markdown("**Thariq Ahmad Baihaqi Adinegara**")
    st.markdown("Purwadhika Digital Technology School - AI Engineering")
    st.markdown("_Data diambil dari IMDb Top 1000 Movies_")
    
    # GitHub repository link
    st.markdown("[Lihat Kode di GitHub](https://github.com/thariqabe666/Project-3-CineBot_Movie_Expert)")
    st.markdown("---")
    # Chat history clear button
    if st.button("Hapus Riwayat Obrolan", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun() # Refresh halaman agar chat kosong


# 4.2: Contoh pertanyaan & callback
# - Tombol contoh untuk mempermudah pengguna memulai (mis. Film mirip Inception, Top 5 gross, dsb.)
# - Fungsi set_user_input sebagai callback untuk menaruh nilai ke st.session_state.user_input.
# Example question buttons
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

# 4.3: Session management & chat history
# - Inisialisasi session_id unik (untuk Langfuse dan tracking sesi).
# - Inisialisasi st.session_state.messages jika belum ada.
# - Tambahkan salam pembuka otomatis bila history kosong.

# Inisialisasi session_id unik untuk Langfuse tracing
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Get user input from chat box or example buttons
chat_input = st.chat_input("Contoh: 'Film mirip Inception' atau 'Top 5 film 2010'")
user_input = chat_input or st.session_state.get("user_input", None)

# Clear button-triggered input after use
if "user_input" in st.session_state and not chat_input:
    del st.session_state.user_input

if "messages" not in st.session_state:
    st.session_state.messages = []

    # --- TAMBAHAN: Salam Pembuka Otomatis ---
    # Tambahkan pesan pertama dari asisten jika history kosong
    if not st.session_state.messages:
        st.session_state.messages.append(
            {"role": "assistant", "content": "Halo! Aku CineBot 🍿 Ada yang bisa kubantu? Kamu bisa tanya rekomendasi film atau data film spesifik!"}
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 1. Convert chat history from dicts to LangChain BaseMessage objects
    from langchain_core.messages import HumanMessage, AIMessage
    langchain_messages = [
        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
        for msg in st.session_state.messages
    ]

    # 2. Variabel untuk menyimpan info proses berpikir
    tool_call_info = None
    full_tool_output = ""
    sql_query_to_display = None
    display_answer = ""
    last_valid_state = None 

    with st.chat_message("assistant"):
        with st.spinner("CineBot sedang mencari jawaban..."):
            
            # Initialize Langfuse callback handler
            langfuse_handler = CallbackHandler()
            
            # 3. Configure Langfuse tracing with metadata
            config = {
                "callbacks": [langfuse_handler],
                "run_name": f"Query: {user_input[:30]}...",
                "metadata": { # Lewatkan atribut di sini
                    "langfuse_session_id": st.session_state.session_id,
                    "langfuse_user_id": st.session_state.session_id, 
                    "langfuse_tags": ["CineBot-v1", "Capstone-Mod3"]
                }
            }            
            
            # 4. Stream agent response with Langfuse configuration
            stream = agent_runnable.stream(
                {"messages": langchain_messages},
                stream_mode="values",
                config=config
            )
            
            for chunk in stream:
                if "messages" in chunk:
                    last_valid_state = chunk 
                    last_message = chunk["messages"][-1]
                    
                    # Capture tool call information when the agent decides to use a tool
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        call = last_message.tool_calls[0]
                        tool_call_info = {
                            "name": call['name'],
                            "args": call['args']
                        }
                        
                    # Capture raw tool output if the message type is 'tool'
                    if hasattr(last_message, "type") and last_message.type == "tool": # LangChain message objects have a .type attribute
                        full_tool_output = last_message.content
            
            # 5. Ambil jawaban akhir (setelah stream selesai)
            if last_valid_state:
                final_answer_object = last_valid_state["messages"][-1]
                display_answer = final_answer_object.content
            else:
                display_answer = "Maaf, terjadi kesalahan."
            
            # 6. Parse SQL query from tool output if SQL tool was used
            if tool_call_info and tool_call_info['name'] == 'get_factual_movie_data':
                if "||SQL_QUERY||" in full_tool_output:
                    parts = full_tool_output.split("||SQL_QUERY||")
                    sql_query_to_display = parts[1]
                else:
                    sql_query_to_display = "Query tidak dapat diekstrak dari tool."
            
            # Display the final answer from the agent.
            # The agent is instructed to format posters as Markdown images within a table.
            # unsafe_allow_html=True is used for robustness in case the agent generates
            # complex markdown or HTML elements.
            st.markdown(display_answer, unsafe_allow_html=True)


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
