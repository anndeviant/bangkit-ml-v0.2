import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PropertyRAG import PropertyRAG
import plotly.express as px

# Load model dan data
@st.cache_resource
def load_keras_models():
    text_model = load_model('text_model.keras')
    numeric_model = load_model('numeric_model.keras')
    return text_model, numeric_model

@st.cache_resource
def load_tfidf_vectorizer():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.warning("File tfidf_vectorizer.pkl tidak ditemukan. Akan membuat TfidfVectorizer baru.")
        return None

@st.cache_data
def load_property_data():
    return pd.read_csv('databaru.csv', encoding='utf-8')

text_model, numeric_model = load_keras_models()
tfidf_vectorizer = load_tfidf_vectorizer()
df = load_property_data()

# Inisialisasi PropertyRAG
rag = PropertyRAG(df, text_model, numeric_model, tfidf_vectorizer)

# Fungsi untuk menampilkan hasil regex
def display_regex_results(query):
    # Gunakan default values atau operator ternary untuk menangani None
    kamar = rag.extract_numeric_requirements(query)[0] or 0
    wc = rag.extract_numeric_requirements(query)[1] or 0
    parkir = rag.extract_numeric_requirements(query)[2] or 0
    max_price = rag.extract_numeric_requirements(query)[3] or float('inf')
    
    st.subheader("Hasil Regex Numeric :")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Jumlah Kamar", kamar if kamar > 0 else "Tidak Terdeteksi")
        st.metric("Jumlah Kamar Mandi", wc if wc > 0 else "Tidak Terdeteksi")
    
    with col2:
        st.metric("Slot Parkir", parkir if parkir > 0 else "Tidak Terdeteksi")
        if max_price != float('inf'):
            st.metric("Harga Maksimal", f"Rp {max_price:,.0f}")
        else:
            st.metric("Harga Maksimal", "Tidak Terdeteksi")
    
    location = rag.find_location(query)
    st.metric("Lokasi yang Terdeteksi", location if location else "Tidak Terdeteksi")
    
# Jika Anda ingin menyimpan TfidfVectorizer yang baru di-fit
if tfidf_vectorizer is None:
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(rag.tfidf_vectorizer, f)
    st.success("TfidfVectorizer baru telah disimpan.")

# Fungsi untuk membuat visualisasi
def create_price_distribution_plot(df):
    fig = px.histogram(df, x='Harga', nbins=50, title='Distribusi Harga Properti')
    fig.update_xaxes(title='Harga (dalam Miliar Rupiah)')
    fig.update_yaxes(title='Jumlah Properti')
    return fig

def create_location_pie_chart(df):
    location_counts = df['Lokasi'].value_counts().head(10)
    fig = px.pie(values=location_counts.values, names=location_counts.index, title='10 Lokasi Teratas')
    return fig

def create_feature_correlation_heatmap(df):
    corr_features = ['Harga_Normalized', 'Kamar_Normalized', 'WC_Normalized', 'Parkir_Normalized', 'Luas_Tanah_Normalized', 'Luas_Bangunan_Normalized']
    corr_matrix = df[corr_features].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title='Korelasi Antar Fitur')
    return fig

# Streamlit app
st.title('Aplikasi Rekomendasi Properti Jogja')

# Sidebar untuk navigasi
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih Halaman", ["Chat", "Distribusi Harga", "Lokasi Populer", "Korelasi Fitur", "Seluruh Data"])

if page == "Chat":
    st.header("Chat Rekomendasi Properti")
    
    # Chat-like interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Masukkan kriteria properti Anda:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = "Berikut adalah rekomendasi properti untuk Anda:"
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

            recommendations = rag.get_recommendations(prompt)
            
            if not recommendations.empty:
                st.dataframe(recommendations)
                
                # Visualisasi rekomendasi
                fig_price = px.scatter(recommendations, x='Luas_Bangunan', y='Harga', 
                                       size='Kamar', hover_name='Judul', 
                                       title='Harga vs Luas Bangunan (ukuran titik menunjukkan jumlah kamar)')
                st.plotly_chart(fig_price)
            else:
                st.warning("Maaf, tidak ada rekomendasi yang sesuai dengan kriteria Anda.")

            display_regex_results(prompt)

elif page == "Distribusi Harga":
    st.header("Distribusi Harga Properti")
    st.plotly_chart(create_price_distribution_plot(df))

elif page == "Lokasi Populer":
    st.header("Lokasi Properti Populer")
    st.plotly_chart(create_location_pie_chart(df))

elif page == "Korelasi Fitur":
    st.header("Korelasi Antar Fitur")
    st.plotly_chart(create_feature_correlation_heatmap(df))

elif page == "Seluruh Data":
    st.header("Jelajahi Seluruh Data Properti")
    
    # Gunakan st.empty() untuk membuat container yang dapat diupdate
    data_container = st.empty()
    
    # Pilihan kolom untuk ditampilkan
    columns_to_show = st.multiselect(
        "Pilih kolom yang ingin ditampilkan", 
        df.columns.tolist(), 
        default=['Judul', 'Lokasi', 'Harga', 'Kamar', 'WC', 'Parkir', 'Luas_Tanah', 'Luas_Bangunan']
    )
    
    # Filter berdasarkan lokasi
    selected_locations = st.multiselect(
        "Filter Lokasi", 
        df['Lokasi'].unique().tolist()
    )
    
    # Filter berdasarkan rentang harga
    min_price = float(df['Harga'].min())
    max_price = float(df['Harga'].max())
    price_range = st.slider(
        "Rentang Harga (Rupiah)", 
        min_price, 
        max_price, 
        (min_price, max_price)
    )
    
    # Filter berdasarkan jumlah kamar
    min_rooms = int(df['Kamar'].min())
    max_rooms = int(df['Kamar'].max())
    rooms_range = st.slider(
        "Jumlah Kamar", 
        min_rooms, 
        max_rooms, 
        (min_rooms, max_rooms)
    )
    
    # Terapkan filter
    filtered_df = df.copy()
    
    if selected_locations:
        filtered_df = filtered_df[filtered_df['Lokasi'].isin(selected_locations)]
    
    filtered_df = filtered_df[
        (filtered_df['Harga'] >= price_range[0]) & 
        (filtered_df['Harga'] <= price_range[1]) &
        (filtered_df['Kamar'] >= rooms_range[0]) & 
        (filtered_df['Kamar'] <= rooms_range[1])
    ]
    
    # Tampilkan dataframe yang difilter dengan kolom yang dipilih
    with data_container.container():
        st.write(f"Jumlah properti yang ditampilkan: {len(filtered_df)}")
        st.dataframe(filtered_df[columns_to_show])
    
    # Tambahkan opsi untuk menampilkan gambar properti
    if st.checkbox("Tampilkan Gambar Properti"):
        selected_property = st.selectbox("Pilih Properti", filtered_df['Judul'])
        image_link = filtered_df[filtered_df['Judul'] == selected_property]['Image_Link'].values[0]
        if pd.notna(image_link):
            st.image(image_link, caption=selected_property)
        else:
            st.write("Gambar tidak tersedia untuk properti ini.")

# Footer
st.markdown("---")
st.markdown("Dibuat dengan ❤️ menggunakan Streamlit")