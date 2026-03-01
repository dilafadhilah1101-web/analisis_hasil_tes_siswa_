import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

st.set_page_config(page_title="Dashboard Analisis Hasil Tes Siswa", layout="wide")

st.title("📊 Dashboard Analisis Hasil Tes 50 Siswa - 20 Soal")
st.markdown("Analisis berbasis data untuk evaluasi hasil pembelajaran")

# Upload file
uploaded_file = st.file_uploader("Upload File Excel Data Siswa", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Preview Data")
    st.dataframe(df)

    # ===============================
    # 1️⃣ Statistik Dasar
    # ===============================
    st.header("1️⃣ Ringkasan Statistik")

    jumlah_siswa = df.shape[0]
    jumlah_soal = df.shape[1]

    df["Total_Skor"] = df.sum(axis=1)
    rata_rata_total = df["Total_Skor"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Jumlah Siswa", jumlah_siswa)
    col2.metric("📝 Jumlah Soal", jumlah_soal)
    col3.metric("📈 Rata-rata Skor Total", round(rata_rata_total, 2))

    # ===============================
    # 2️⃣ Rata-rata per Soal
    # ===============================
    st.header("2️⃣ Analisis Rata-rata per Soal")

    rata_per_soal = df.iloc[:, :-1].mean()

    fig, ax = plt.subplots()
    rata_per_soal.plot(kind="bar", ax=ax)
    plt.xticks(rotation=90)
    plt.title("Rata-rata Skor per Soal")
    st.pyplot(fig)

    soal_terendah = rata_per_soal.idxmin()
    soal_tertinggi = rata_per_soal.idxmax()

    st.success(f"Soal Termudah: {soal_tertinggi}")
    st.error(f"Soal Tersulit: {soal_terendah}")

    # ===============================
    # 3️⃣ Korelasi
    # ===============================
    st.header("3️⃣ Korelasi Antar Soal")

    corr = df.iloc[:, :-1].corr()

    fig2, ax2 = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

    # ===============================
    # 4️⃣ Regresi Linear
    # ===============================
    st.header("4️⃣ Analisis Regresi Linear")

    X = df.iloc[:, :-1]
    y = df["Total_Skor"]

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)

    st.metric("📈 Nilai R²", round(r2, 3))

    koef = pd.Series(model.coef_, index=X.columns)
    faktor_dominan = koef.abs().idxmax()

    st.write("🔑 Faktor (Soal) Paling Berpengaruh:", faktor_dominan)

    # ===============================
    # 5️⃣ Segmentasi Siswa
    # ===============================
    st.header("5️⃣ Segmentasi Siswa (Clustering)")

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df["Total_Skor"], y=df["Cluster"], hue=df["Cluster"], palette="Set1")
    st.pyplot(fig3)

    st.success("Segmentasi berhasil – siap untuk rekomendasi tindak lanjut pembelajaran")
