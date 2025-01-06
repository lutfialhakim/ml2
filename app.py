import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import export_graphviz
import graphviz
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load models from the specified paths
fish_model_svm = pickle.load(open('fish/svm_fish.pkl', 'rb'))
#fish_scaler_svm = pickle.load(open('fish/svm_scaler_fish.pkl', 'rb'))
fish_model_rfc = pickle.load(open('fish/randomforest_fish.pkl', 'rb'))
fish_model_perceptron = pickle.load(open('fish/prcp_fish.pkl', 'rb'))  # Load Perceptron model
fish_model_prcp = pickle.load(open('fish/prcp_fish.pkl', 'rb'))

fruit_model_svm = pickle.load(open('fruit/svm_fruit.pkl', 'rb'))
#fruit_scaler_svm = pickle.load(open('fruit/svm_scaler_fruit.pkl', 'rb'))
fruit_model_rfc = pickle.load(open('fruit/randomforest_fruit.pkl', 'rb'))
fruit_model_perceptron = pickle.load(open('fruit/prcp_fruit.pkl', 'rb'))  # Load Perceptron model
fruit_model_prcp = pickle.load(open('fruit/prcp_fruit.pkl', 'rb'))

pumpkin_model_svm = pickle.load(open('pumpkin/svm_pumpkin.pkl', 'rb'))
#pumpkin_scaler_svm = pickle.load(open('pumpkin/svm_scaler_pumpkin.pkl', 'rb'))
pumpkin_model_rfc = pickle.load(open('pumpkin/randomforest_pumpkin.pkl', 'rb'))
pumpkin_model_perceptron = pickle.load(open('pumpkin/prcp_pumpkin.pkl', 'rb'))  # Load Perceptron model
pumpkin_model_prcp = pickle.load(open('pumpkin/prcp_pumpkin.pkl', 'rb'))

wine_model_kmeans = pickle.load(open('kmeans/kmeans_wine2.pkl', 'rb'))

# Page title
st.title('Prediksi Machine Learning')

# Select classification category
st.write("### Pilih Kategori")
option = st.selectbox("Klasifikasi:", ("Fish", "Fruit", "Pumpkin", "Wine"))

# Select algorithm
if option == "Wine":
            st.write("### Pilih Algoritma")
            algorithm = "K-Means"
            try:
                # Load model .pkl
                with open("kmeans/kmeans_wine.pkl", "rb") as file:
                    kmeans_model = pickle.load(file)

                # Simulasikan dataset
                wine_data = pd.DataFrame({
                    "alcohol": np.random.uniform(10, 15, 200),
                    "total_phenols": np.random.uniform(0.1, 5, 200),
                })
                wine_features = wine_data.values

                # Input untuk jumlah maksimal K
                max_k = st.slider("Pilih Maksimal K", min_value=2, max_value=20, value=17)

                # Tombol untuk menampilkan Elbow Method
                if st.button("Tampilkan Grafik Elbow Method"):
                    # Hitung SSE untuk setiap nilai K
                    sse = []
                    for k in range(1, max_k + 1):
                        kmeans = KMeans(n_clusters=k, random_state=10)
                        kmeans.fit(wine_features)
                        sse.append(kmeans.inertia_)

                    # Plot grafik Elbow Method
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, max_k + 1), sse, marker='o')
                    plt.xlabel("Jumlah Kluster (K)")
                    plt.ylabel("Sum of Squared Errors (SSE)")
                    plt.title("Elbow Method untuk Menentukan Nilai Optimal K")
                    plt.grid(True)

                    # Tampilkan grafik di Streamlit
                    st.pyplot(plt)

            except FileNotFoundError:
                st.error("Model kmean_wine.pkl tidak ditemukan! Pastikan file ada di direktori yang sama.")
       
else:
    st.write("### Pilih Algoritma")
    algorithm = st.selectbox("Algoritma:", ("SVM", "Random Forest", "Perceptron"))  # Added Perceptron

st.markdown("---")

# Dictionaries for fish, fruit, and pumpkin types
fish_types = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conchonius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}

fruit_types = {0: "Grapefruit", 1: "Orange"}

pumpkin_types = {0: "Çerçevelik", 1: "Ürgüp Sivrisi"}

# Input form based on category
with st.form(key='prediction_form'):
    if option == "Fish":
        st.write("### Masukkan Data Ikan")
        weight = st.number_input('Berat Ikan (dalam gram)', min_value=0.0, format="%.2f")
        length = st.number_input('Panjang Ikan (dalam cm)', min_value=0.0, format="%.2f")
        height = st.number_input('Rasio Berat & Panjang (dalam cm)', min_value=0.0, format="%.2f")
        
        submit = st.form_submit_button(label='Prediksi Jenis Ikan')
        
        if submit:
            input_data = np.array([weight, length, height]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = fish_model_svm.predict(input_data)
            elif algorithm == "Random Forest":
                prediction = fish_model_rfc.predict(input_data)
                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = fish_model_rfc.estimators_[0]
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Weight", "Length", "Height"],
                    class_names=list(fish_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    max_depth=3
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)
            else:  # Perceptron
                prediction = fish_model_perceptron.predict(input_data)

            fish_result = fish_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Ikan: {fish_result}")

    elif option == "Fruit":
        st.write("### Masukkan Data Buah")
        diameter = st.number_input('Diameter Buah (dalam cm)', min_value=0.0, format="%.2f")
        weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
        red = st.slider('Skor Warna Buah Merah', 0, 255, 0)
        green = st.slider('Skor Warna Buah Hijau', 0, 255, 0)
        blue = st.slider('Skor Warna Buah Biru', 0, 255, 0)
        
        submit = st.form_submit_button(label='Prediksi Jenis Buah')
        
        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = fruit_model_svm.predict(input_data)
            elif algorithm == "Random Forest":
                prediction = fruit_model_rfc.predict(input_data)
                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = fruit_model_rfc.estimators_[0]
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Diameter", "Weight", "Red", "Green", "Blue"],
                    class_names=list(fruit_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    max_depth=3
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)
            else:  # Perceptron
                prediction = fruit_model_perceptron.predict(input_data)

            fruit_result = fruit_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Buah: {fruit_result}")
  
    elif option == "Pumpkin":
        st.write("### Masukkan Data Labu")
        area = st.number_input('Area (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        perimeter = st.number_input('Keliling (dalam cm)', min_value=0.0, format="%.2f")
        major_axis_length = st.number_input('Panjang Sumbu Mayor (dalam cm)', min_value=0.0, format="%.2f")
        minor_axis_length = st.number_input('Panjang Sumbu Minor (dalam cm)', min_value=0.0, format="%.2f")
        convex_area = st.number_input('Area Cembung (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        equiv_diameter = st.number_input('Diameter Ekivalen (dalam cm)', min_value=0.0, format="%.2f")
        eccentricity = st.number_input('Eksentrisitas', min_value=0.0, format="%.2f")
        solidity = st.number_input('Kepadatan', min_value=0.0, format="%.2f")
        extent = st.number_input('Ekstensi', min_value=0.0, format="%.2f")
        roundness = st.number_input('Kebulatan', min_value=0.0, format="%.2f")
        aspect_ratio = st.number_input('Rasio Aspek', min_value=0.0, format="%.2f")
        compactness = st.number_input('Kompak', min_value=0.0, format="%.2f")

        submit = st.form_submit_button(label='Prediksi Jenis Labu')
    
        
        if submit:
            input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area, equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, compactness]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = pumpkin_model_svm.predict(input_data)
            elif algorithm == "Random Forest":
                prediction = pumpkin_model_rfc.predict(input_data)
                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = pumpkin_model_rfc.estimators_[0]
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Area", "Perimeter", "Major Axis Length", "Minor Axis Length", "Convex Area", "Equiv Diameter", "Eccentricity", "Solidity", "Extent", "Roundness", "Aspect Ratio", "Compactness"],
                    class_names=list(pumpkin_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True,
                     max_depth=3
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)
            else:  # Perceptron
                prediction = pumpkin_model_perceptron.predict(input_data)

            pumpkin_result = pumpkin_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Labu: {pumpkin_result}")
