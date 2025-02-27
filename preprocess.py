import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import re

def normalize_text(text):
    # Normalizar URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Normalizar menciones de usuario (@usuario)
    text = re.sub(r'@\w+', '[USER]', text)
    
    # Normalizar hashtags (#hashtag)
    text = re.sub(r'#\w+', '[HASHTAG]', text)
    
    return text

def preprocess_data(text_data:str)->pd.DataFrame:

    # Remove STOPWORDS
    text_data = text_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (ENGLISH_STOP_WORDS)]))

    normalized_text= normalize_text(text_data)

    # Preprocess the data
    return normalized_text
