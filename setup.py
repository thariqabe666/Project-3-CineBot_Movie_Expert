import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Muat environment variables dari file .env
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# 1. Tentukan Path dan Nama
csv_path = 'data/imdb_top_1000_cleaned.csv'
db_path = 'sqlite:///movies.db' # Ini akan membuat file 'movies.db'
qdrant_collection_name = 'imdb_movies'

# 2. Load Data dari CSV
try:
    df = pd.read_csv(csv_path)
    print(f"Data CSV '{csv_path}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File CSV tidak ditemukan di '{csv_path}'.")
    exit()

# === BAGIAN A: SETUP DATABASE SQL (UNTUK TOOL 2) ===
print("\nMemulai setup database SQL...")
try:
    # Buat koneksi engine ke file database SQLite
    engine = create_engine(db_path)
    
    # Simpan DataFrame ke database SQL sebagai tabel bernama 'movies'
    # if_exists='replace' berarti akan menimpa tabel jika sudah ada
    df.to_sql('movies', engine, if_exists='replace', index=False)
    
    print(f"Database SQL 'movies.db' dan tabel 'movies' berhasil dibuat.")
except Exception as e:
    print(f"ERROR saat membuat database SQL: {e}")
    exit()

# === BAGIAN B: SETUP DATABASE VECTOR (UNTUK TOOL 1) ===
print("\nMemulai setup database vector (Qdrant)...")
try:
    # Inisialisasi model embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # Gabungkan kolom teks untuk embedding yang lebih kaya
    # Ini adalah "text_for_embedding" yang kita diskusikan
    df['text_for_embedding'] = (
        "Judul: " + df['Series_Title'] + "; " +
        "Genre: " + df['Genre'] + "; " +
        "Sutradara: " + df['Director'] + "; " +
        "Pemeran: " + df['Star1'] + ", " + df['Star2'] + ", " + df['Star3'] + "; " +
        "Sinopsis: " + df['Overview']
    )

    # Konversi DataFrame Pandas menjadi daftar Document LangChain
    # Kita juga sertakan metadata penting lainnya
    documents = [
        Document(
            page_content=row['text_for_embedding'],
            metadata={
                'id': i,
                'title': row['Series_Title'],
                'year': row['Released_Year'],
                'rating': row['IMDB_Rating'],
                'genre': row['Genre'],
                'poster': row['Poster_Link']
            }
        ) for i, row in df.iterrows()
    ]

    # --- PERBAIKAN: Gunakan batching untuk menghindari timeout ---
    # Inisialisasi client Qdrant
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60  # Tambahkan timeout yang lebih lama (dalam detik)
    )

    # Buat QdrantVectorStore dari client yang sudah ada
    # Ini memungkinkan kita untuk menggunakan metode .add_documents() untuk batching
    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection_name,
        embedding=embeddings
    )

    # Hapus dan buat ulang koleksi (sama seperti force_recreate=True)
    client.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=client.get_collection(collection_name=qdrant_collection_name).config.params.vectors
    )
    print(f"Koleksi '{qdrant_collection_name}' telah dikosongkan dan siap diisi ulang.")

    # Tentukan ukuran batch
    batch_size = 50
    total_docs = len(documents)

    # Loop dan upload dokumen dalam batch
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        qdrant_store.add_documents(batch)
        print(f"Mengunggah batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}... ({min(i + batch_size, total_docs)}/{total_docs} dokumen)")
    # --- AKHIR PERBAIKAN ---

    print(f"Koleksi '{qdrant_collection_name}' di Qdrant berhasil dibuat/diisi.")

except Exception as e:
    print(f"ERROR saat setup Qdrant: {e}")
    exit()

print("\n=== SETUP SELESAI ===")
print("Kamu sekarang siap untuk menjalankan 'main.py'.")