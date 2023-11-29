
import gensim
import pandas as pd
import numpy as np
import ast

class Word2VecModel:

    def __init__(self):
        self.model = None

    def train_word2vec(self, df, vector_size=400, window=5, min_count=1, workers=4, epochs=10, sg=0, hs=0):

        # Convert string representation of list into actual list only if it's a string
        def convert_to_list(value):
            if isinstance(value, str):
                try:
                    return ast.literal_eval(value)
                except:
                    return [value]  # return the string itself as a list with one item if there's an error
            return value

        df['tokens'] = df['tokens'].apply(convert_to_list)

        # Now we can process the tokens as before
        self.sentences = [[token.lower() for token in tokens] for tokens in df['tokens']]
        # self.sentences = df['tokens'].tolist()
        self.model = gensim.models.word2vec.Word2Vec(self.sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs, sg=sg, hs=hs)
        # self.model.train(self.sentences, total_examples=len(self.sentences), epochs=epochs)

    def get_vector(self, word):
        try:
            return self.model.wv[word]
        except KeyError:
            return None

    def save_embeddings_to_txt(self, path="word2vec/word2vec_embeddings.txt"):
        # Get the model's vector size
        vector_size = self.model.vector_size

        # Initialize a list to store valid embeddings
        valid_embeddings = []

        for word in self.model.wv.index_to_key:
            # Remove extra spaces from the word
            cleaned_word = ' '.join(word.split())
            
            # Skip if the cleaned word is an empty string or contains whitespace
            if not cleaned_word or ' ' in cleaned_word or '\t' in cleaned_word:
                continue

            vector = self.model.wv[word]
            # Check if vector length is correct
            if len(vector) == vector_size:
                vector_str_list = list(map(str, vector))
                # Convert the vector string list to a single string and store it
                vector_str = ' '.join(vector_str_list)
                valid_embeddings.append(f"{cleaned_word} {vector_str}")
            else:
                print(f"Warning: Vector size for word '{word}' is not {vector_size}. Skipping.")

        # Get the updated vocabulary size
        updated_vocab_size = len(valid_embeddings)

        with open(path, 'w', encoding='utf-8') as f:
            # Write the updated vocabulary size and vector size to the first line
            f.write(f"{updated_vocab_size} {vector_size}\n")

            # Write all the valid embeddings to the file
            for embedding in valid_embeddings:
                f.write(f"{embedding}\n")

        print(f"Embeddings saved to {path}")
