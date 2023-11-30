import os
import time
import ast
import gensim

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Importing our created classes
from Preprocessing import Preprocessing
from Word2vec_model import Word2VecModel
from Model import Model

# Initialize session state for DataFrame
# if 'df' not in st.session_state:
#     st.session_state.df = None

class Gui:
    def __init__(self):
        self.df = None
        self.preprocessing_modified = None
        self.word2vec = None
        # Muat tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # Muat model
        # self.model_path = load_model('model/model.h5')
        # Set model path
        model_path = 'model/model.h5'

        # Cek jika model ada
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        
    def run(self):
        st.title('Language Identification Using Long Short Term Memory Algorithm')
        # self.train_lstm()
        self.load_dataset()
        self.preprocess_data()
        
        if st.button('Word2Vec'):
            self.word2vec_data()
        if st.button('Train LSTM Model'):
            self.train_lstm_model()

        # Membuat text area untuk input user
        input_text = st.text_input("Input Teks")

        # Membuat tombol prediksi
        if st.button('Prediksi'):
            if input_text:  # Cek jika ada input dari pengguna
                # Jalankan fungsi prediksi dan tampilkan hasilnya
                language_label = self.make_prediction(input_text)
                st.success(f"Bahasa yang diprediksi adalah: {language_label}")
            else:
                st.error("Silakan masukkan teks untuk diprediksi.")

            
    def load_dataset(self):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            self.df = pd.read_csv(uploaded_file)
            st.write(self.df)
            # self.preprocessing_modified = Preprocessing(self.df)

    def preprocess_data(self):
        if st.button("Preprocess Data"):
            progress_bar = st.progress(0)
            for i in range(100):
                # Increment progress
                progress_bar.progress(i + 1)
                time.sleep(0.01)
            # self.df = self.actual_preprocessing()
            self.preprocessing_modified = Preprocessing(self.df)
            st.session_state.df = self.preprocessing_modified.process_text()
            st.write('Preprocessed Data:')
            st.write(self.df)
            st.write(f"Processed {len(self.df)} sentences.")
            # self.preprocessing_modified = Preprocessor(self.df)
    
    def word2vec_data(self):
        # if st.button("Word2Vec Data"):
        #     self.word2vec = Word2VecModel()
        #     self.word2vec.train_word2vec(self.data)
        #     # st.session_state.word2vec_model = word2vec
        #     st.session_state.word2vec_model = self.word2vec
        #     st.success("Word2Vec model trained and saved!")

        # Read the preprocessed data from the temporary file
        if self.df is not None:
            self.df = pd.read_csv('preprocessed_data/temp_preprocessed_data.csv')
            
            self.word2vec = Word2VecModel()
            
            self.word2vec.train_word2vec(self.df)
            self.word2vec.save_embeddings_to_txt()
            st.session_state.word2vec_model = self.word2vec
            # Display some information about the word2vec object
            vocab_size = len(self.word2vec.model.wv)
            st.write(f"Word2Vec Model Info:")
            st.write(f"Vocabulary Size: {vocab_size}")
    
            st.write("Word2Vec model has been trained!")
            sample_words = ["muncul", "pertama", "adalah", "ketika", "cium"]
            for word in sample_words:
                vector = self.word2vec.get_vector(word)
                if vector is not None:
                    st.write(f"Vector for word '{word}':", vector[:5])  # Displaying only the first 5 values of the vector for simplicity
                else:
                    st.write(f"Word '{word}' not found in vocabulary.")

        else:
            st.write("Please preprocess the data first.")

        if self.word2vec:
            st.write("self.word2vec is initialized!")
            try:
                st.write("Embedding matrix obtained successfully!")
            except Exception as e:
                st.write("Error when trying to get embedding matrix:", str(e))
        else:
            st.write("self.word2vec is not initialized!")

    def train_lstm_model(self):
        # Load the preprocessed data
        df_preprocessed = pd.read_csv('preprocessed_data/temp_preprocessed_data.csv')
        
        # Create an instance of the Model and train it with the preprocessed data
        model_instance = Model()
        model, report, accuracy = model_instance.buildModel(df_preprocessed)
    
        st.write("Training completed!")
        
        # Convert the classification report dictionary to a DataFrame
        report_df = pd.DataFrame(report).transpose()
        
        # Display the classification report
        st.write(f"Accuracy: {accuracy:.4f}")
        st.dataframe(report_df)

        # Membaca confusion matrix yang disimpan
        conf_matrix_df = pd.read_csv('confusion_matrix.csv', index_col=0)

        # Menampilkan confusion matrix sebagai tabel
        st.write("Confusion Matrix")
        st.dataframe(conf_matrix_df)

        # Menampilkan confusion matrix sebagai heatmap
        st.write("Confusion Matrix Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', ax=ax)  # fmt='d' untuk angka integer
        st.pyplot(fig)

        # Read the metrics CSV file
        metrics_tfpn_df = pd.read_csv('metrics_tfpn.csv', index_col='Label')

        # Display the DataFrame in the GUI
        st.write("Metrics Table (True Positives, True Negatives, False Positives, and False Negatives):")
        st.table(metrics_tfpn_df)

    def make_prediction(self, input_text):
        
        # Lakukan tokenisasi dan padding pada input text
        sequence = self.tokenizer.texts_to_sequences([input_text])
        padded_sequence = pad_sequences(sequence, maxlen=100)  # Sesuaikan dengan nilai yang digunakan saat pelatihan model
        
        # Lakukan prediksi
        prediction = self.model_path.predict(padded_sequence)
        
        # Definisikan label di dalam fungsi yang sama
        labels = ['Arabic', 'Chinese', 'Dutch', 'English', 'French', 'Indonesian', 'Japanese', 'Korean', 'Russian', 'Spanish']
        
        # Dapatkan indeks dengan nilai tertinggi (kelas prediksi)
        predicted_index = np.argmax(prediction)
        
        # Kembalikan label yang sesuai dengan indeks prediksi
        return labels[predicted_index]

if __name__ == '__main__':
    app = Gui()
    app.run()
