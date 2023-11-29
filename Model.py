import numpy as np
import ast  # To convert string representation of list to actual list
from gensim.models import KeyedVectors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History

class Model:
# # Define hyperparameter ranges
# batch_sizes = [32, 64, 128, 256]
# dropouts = [0.2, 0.3, 0.4, 0.5]
# epochs_list = [10, 15, 20]
# hidden_units_list = [32, 64, 128, 256, 512]
# recurrent_dropouts = [0.2, 0.3, 0.4, 0.5]
    def __init__(self, dropout=0.3, hidden_units=64, recurrent_dropout=0.2, epochs=15, batch_size=32):
    # def __init__(self, dropout=0.3, hidden_units=32, recurrent_dropout=0.3, epochs=10, batch_size=64):
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
    def buildModel(self, data_preprocessed):
        data = data_preprocessed
        
        # Convert string representation of tokens to actual list
        # data['tokens'] = data['tokens'].apply(ast.literal_eval)

        # Convert string representation of list into actual list only if it's a string
        def convert_to_list(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except:
                    return [value]  # return the string itself as a list with one item if there's an error
            return value

        data['tokens'] = data['tokens'].apply(convert_to_list)

        # Now we can process the tokens as before
        self.sentences = [[token.lower() for token in tokens] for tokens in data['tokens']]
        
        # One-hot encoding for the languages
        labels = pd.get_dummies(data['language']).values
        
        # Split data into training and test sets
        tokens_train, tokens_test, labels_train, labels_test = train_test_split(data['tokens'], labels, test_size=0.2, random_state=42)
        
        # Tokenizing the data
        tokenizer = Tokenizer(oov_token='<UNK>', filters='')
        # np.save("tokenized.npy",data['tokens'])
        tokenizer.fit_on_texts(tokens_train)
        # # Save the tokenizer to a file
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('-------Tokenizing selesai-------')
        
        # Load Word2Vec embeddings
        word2vec_model = KeyedVectors.load_word2vec_format('word2vec/word2vec_embeddings.txt', binary=False, encoding='utf-8')
        vector_size = word2vec_model.vector_size

        # Create the embedding matrix
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, vector_size))
        for word, i in tokenizer.word_index.items():
            if word in word2vec_model:
                embedding_matrix[i] = word2vec_model[word]
        
        # Padding the sequences
        def get_sequences(token, data, max_train=100):
            sequences = token.texts_to_sequences(data)
            padded_sequences = pad_sequences(sequences, maxlen=max_train)
            return padded_sequences

        padded_train_sequences = get_sequences(tokenizer, tokens_train)
        padded_test_sequences = get_sequences(tokenizer, tokens_test)
        
        # Building the LSTM model with Word2Vec embeddings
        num_unique_languages = labels.shape[1]
        model = Sequential()
        model.add(Embedding(vocab_size, vector_size, weights=[embedding_matrix], input_length=100, trainable=False))
        model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
        model.add(LSTM(self.hidden_units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(num_unique_languages, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = History()

        # Training the model
        model.fit(padded_train_sequences, np.array(labels_train), validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size, callbacks=[history])
        
        # Evaluating the model
        predicted_labels = model.predict(padded_test_sequences)
        predicted_labels_rounded = np.argmax(predicted_labels, axis=1)
        true_labels = np.argmax(labels_test, axis=1)

        # Calculate metrics for each label
        accuracy = accuracy_score(true_labels, predicted_labels_rounded)
        f1 = f1_score(true_labels, predicted_labels_rounded, average=None)
        precision = precision_score(true_labels, predicted_labels_rounded, average=None)
        recall = recall_score(true_labels, predicted_labels_rounded, average=None)

        # Generate a classification report
        label_names = data['language'].unique() # This assumes your dataset includes a 'language' column with all labels
        report = classification_report(true_labels, predicted_labels_rounded, target_names=label_names, output_dict=True)

        # Print the metrics
        print(f"Accuracy: {accuracy:.2f}")
        for label in label_names:
            idx = list(label_names).index(label)
            print(f"{label} Precision: {precision[idx]:.2f}")
            print(f"{label} Recall: {recall[idx]:.2f}")
            print(f"{label} F1-Score: {f1[idx]:.2f}")

        model.save('model/model.h5')
        print("model saved successfully")


        # Mengkonversi label prediksi yang dikembalikan oleh model ke bentuk diskret (misalnya, kelas terprediksi)
        predicted_labels_rounded = np.argmax(predicted_labels, axis=1)

        # Menghitung confusion matrix
        # Anda perlu mengganti 'true_labels' dengan variabel yang sesuai yang berisi label sebenarnya
        conf_matrix = confusion_matrix(true_labels, predicted_labels_rounded)

        # Calculate TP, TN, FP, and FN without using a separate function
        FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
        FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
        TP = np.diag(conf_matrix)
        TN = conf_matrix.sum() - (FP + FN + TP)

        # Now you can create a DataFrame from these values if you need to
        metrics_df = pd.DataFrame({
            'Label' : label_names,
            'True Positives': TP,
            'True Negatives': TN,
            'False Positives': FP,
            'False Negatives': FN
        }).set_index('Label')

        # Save this DataFrame to a CSV file if needed
        metrics_df.to_csv('confusion_matrix/metrics_tfpn.csv')

        # You might also want to log this information, display it in the console, or handle it however you see fit
        print(metrics_df)

        # Mengubah confusion matrix menjadi DataFrame untuk memudahkan penyimpanan dan visualisasi
        conf_matrix_df = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)

        # Menyimpan confusion matrix sebagai CSV
        conf_matrix_df.to_csv('confusion_matrix/confusion_matrix.csv', index=True)

        # Membuat grafik akurasi
        plt.figure()
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Accuracy Chart')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy/Accuracy_graphic.png')
        plt.close()

        # Membuat grafik loss
        plt.figure()
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Chart')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('accuracy/Loss_graphic.png')
        plt.close()

        return model, report, accuracy
