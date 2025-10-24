# === BAGIAN 1: IMPORTS & ENVIRONMENT ===
# 1.1: Imports utama
# - Library untuk file I/O, data processing, SQL, embeddings, Qdrant, dan dotenv.
# - Jangan ubah import: mereka diperlukan untuk proses setup.
import os
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# 1.2: Load environment variables
# - Prioritas: file .env lokal. Variabel penting:
#   * OPENAI_API_KEY: untuk embedding OpenAI
#   * QDRANT_URL / QDRANT_API_KEY: koneksi ke Qdrant
# - Tujuan: terpisah antara konfigurasi (secrets) dan kode.
load_dotenv() 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# === BAGIAN 2: PATHS & KONSTANTA ===
# 2.1: File / resource paths
# - csv_path: sumber data IMDb yang sudah dibersihkan
# - db_path: SQLite URI yang akan menghasilkan file movies.db
# - qdrant_collection_name: nama koleksi tempat vector akan disimpan
csv_path = 'data/imdb_top_1000_cleaned.csv'
db_path = 'sqlite:///movies.db' # Ini akan membuat file 'movies.db'
qdrant_collection_name = 'imdb_movies'

# === BAGIAN 3: LOAD DATA CSV ===
# 3.1: Tujuan
# - Memuat CSV sebagai DataFrame pandas untuk diproses lebih lanjut.
# 3.2: Error handling
# - Jika file tidak ditemukan, hentikan proses dengan pesan yang jelas.
try:
    df = pd.read_csv(csv_path)
    print(f"Data CSV '{csv_path}' berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File CSV tidak ditemukan di '{csv_path}'.")
    exit()

# === BAGIAN 4: SETUP DATABASE SQL (UNTUK TOOL SQL) ===
# 4.1: Tujuan
# - Menyimpan DataFrame ke SQLite agar tool SQL dapat dijalankan terhadap tabel 'movies'.
# 4.2: Pendekatan
# - Gunakan create_engine dari SQLAlchemy dan df.to_sql(if_exists='replace').
# 4.3: Catatan
# - if_exists='replace' menimpa tabel lama; ini berguna untuk development/refresh data.
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

# === BAGIAN 5: SETUP VECTOR DATABASE (QDRANT) ===
# 5.1: Tujuan utama
# - Membuat koleksi vector di Qdrant berisi embedded documents dari dataset.
# - Dokumen akan digunakan untuk RAG / similarity search (Tool rekomendasi).
# 5.2: Langkah besar
# - Inisialisasi embeddings
# - Siapkan teks yang akan di-embed (gabungan kolom relevan)
# - Konversi baris menjadi Document with metadata
# - Inisialisasi Qdrant client & VectorStore
# - (Opsional tapi direkomendasikan) Recreate collection secara aman
# - Unggah dokumen secara batching untuk mengurangi timeout dan beban
print("\nMemulai setup database vector (Qdrant)...")
try:
    # 5.3: Inisialisasi model embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # 5.4: Siapkan teks untuk embedding
    # - Gabungkan field penting (judul, genre, director, cast, overview)
    # - Tujuan: memberikan konteks yang kaya untuk similarity search
    df['text_for_embedding'] = (
        "Judul: " + df['Series_Title'] + "; " +
        "Genre: " + df['Genre'] + "; " +
        "Sutradara: " + df['Director'] + "; " +
        "Pemeran: " + df['Star1'] + ", " + df['Star2'] + ", " + df['Star3'] + "; " +
        "Sinopsis: " + df['Overview']
    )

    # 5.5: Konversi ke Document (LangChain)
    # - Sertakan metadata yang berguna (title, year, rating, genre, poster)
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

    # 5.6: Inisialisasi Qdrant client & VectorStore
    # - Tambahkan timeout lebih besar untuk mengurangi kemungkinan kegagalan saat upload
    # - Gunakan QdrantVectorStore wrapper untuk kemudahan API (add_documents, similarity_search, dsb.)
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=60  # Tambahkan timeout yang lebih lama (dalam detik)
    )

    qdrant_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection_name,
        embedding=embeddings
    )

    # 5.7: (Opsional) Recreate collection secara aman
    # - Tujuan: pastikan koleksi kosong sebelum upload ulang
    # - Catatan: panggilan ini asumsi koleksi sudah pernah dibuat; jika belum, behavior client harus dicek
    client.recreate_collection(
        collection_name=qdrant_collection_name,
        vectors_config=client.get_collection(collection_name=qdrant_collection_name).config.params.vectors
    )
    print(f"Koleksi '{qdrant_collection_name}' telah dikosongkan dan siap diisi ulang.")

    # 5.8: Upload document secara batching
    # - Alasan batching: mengurangi timeout, memori, dan beban jaringan
    # - Pilih batch_size sesuai kualitas koneksi dan quota API
    batch_size = 50
    total_docs = len(documents)

    for i in range(0, total_docs, batch_size):
        batch = documents[i:i+batch_size]
        qdrant_store.add_documents(batch)
        print(f"Mengunggah batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}... ({min(i + batch_size, total_docs)}/{total_docs} dokumen)")
    # --- AKHIR UPLOAD BATCH ---

    print(f"Koleksi '{qdrant_collection_name}' di Qdrant berhasil dibuat/diisi.")

except Exception as e:
    # 5.9: Error handling di tahap vector setup
    # - Cetak pesan error yang jelas supaya mudah debug (mis. credentials, network, quota)
    print(f"ERROR saat setup Qdrant: {e}")
    exit()

# === BAGIAN 6: PENUTUP / CATATAN PENTING ===
# 6.1: Tanda bahwa setup selesai
# 6.2: Instruksi singkat: jalankan main.py setelah setup sukses
print("\n=== SETUP SELESAI ===")
print("Kamu sekarang siap untuk menjalankan 'main.py'.")