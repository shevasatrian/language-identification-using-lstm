import string
import pandas as pd
import ast

import nltk
from nltk.stem import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import nltk
from nltk.tokenize import word_tokenize
from tinysegmenter import TinySegmenter  # Need to install TinySegmenter
import jieba  # Need to install jieba

class Preprocessing:
    def __init__(self, file_dataset):
        if not file_dataset.empty:
            self.df = file_dataset

    def _cleaning(self, text):
        # Case folding
        text = text.lower() 
        # Trim text
        text = text.strip()
        # Remove punctuations, special characters, and double whitespace
        text = re.compile('<.*?>').sub('', text) 
        text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
        text = re.sub('\s+', ' ', text)
        # Number removal
        text = re.sub(r'\[[0-9]*\]', ' ', text) 
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        # Remove number and whitespaces
        text = re.sub(r'\d', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub('  +', ' ', text)  # Remove extra spaces
        text = text.strip() #hapus spasi dari sisi kiri-kanan teks
        text = re.sub('\s+', ' ', text)
        return text

    def _tokenize(self, text, language):
        if language == 'Japanese':
            segmenter = TinySegmenter()  # Uncomment if TinySegmenter is installed
            tokens = segmenter.tokenize(text)
            # tokens = []  # Placeholder, remove this when using TinySegmenter
        elif language == 'Chinese':
            tokens = jieba.lcut(text)  # Uncomment if jieba is installed
            # tokens = []  # Placeholder, remove this when using jieba
        else:
            tokens = word_tokenize(text)
        return tokens

    def process_text(self):
        self.df['Text'] = self.df['Text'].apply(self._cleaning)
        self.df['tokens'] = self.df.apply(lambda row: self._tokenize(row['Text'], row['language']), axis=1)
        # self.df.to_csv("temp_preprocessed_data.csv", index=False)  
        
        # Apply stemming based on the language
        def apply_stemming(row):
            language = row['language']
            text = row['tokens']
            if language == 'Indonesian':
                return indonesian_stemming(text)
            elif language == 'English':
                return english_stemming(text)
            elif language == 'Arabic':
                return arabic_stemming(text)
            elif language == 'Dutch':
                return dutch_stemming(text)
            elif language == 'French':
                return french_stemming(text)
            elif language == 'Russian':
                return russian_stemming(text)
            elif language == 'Spanish':
                return spanish_stemming(text)
            else:
                return text
        
        self.df['stemmed_tokens'] = self.df.apply(apply_stemming, axis=1)

        
        self.df['tokens'] = self.df['stemmed_tokens']
        self.df.drop(columns=['stemmed_tokens'], inplace=True)
        # self.df['tokens'] = self.df.apply(lambda row: self._tokenize(row['Text'], row['language']), axis=1)
        self.df.to_csv("preprocessed_data/temp_preprocessed_data.csv", index=False)  

        return self.df

    # Other methods from the previous Preprocessor class...

    # Indonesian stemming
def indonesian_stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text]

# English stemming
def english_stemming(text):
    english_stemmer = SnowballStemmer("english")
    return [english_stemmer.stem(word) for word in text]

# Arabic stemming
def arabic_stemming(text):
    arabic_stemmer = SnowballStemmer("arabic")
    return [arabic_stemmer.stem(word) for word in text]

# Dutch stemming
def dutch_stemming(text):
    dutch_stemmer = SnowballStemmer("dutch")
    return [dutch_stemmer.stem(word) for word in text]

# French stemming
def french_stemming(text):
    french_stemmer = SnowballStemmer("french")
    return [french_stemmer.stem(word) for word in text]

# Russian stemming
def russian_stemming(text):
    russian_stemmer = SnowballStemmer("russian")
    return [russian_stemmer.stem(word) for word in text]

# Spanish stemming
def spanish_stemming(text):
    spanish_stemmer = SnowballStemmer("spanish")
    return [spanish_stemmer.stem(word) for word in text]
