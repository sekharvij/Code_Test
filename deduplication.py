import pandas as pd
import numpy as np
import re
import emoji
from rapidfuzz import fuzz

try:
    import faiss
    import tensorflow_hub as hub
except ImportError:
    faiss = None
    hub = None

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = emoji.replace_emoji(text, replace='')
    return text

def clean_dataframe(df):
    return df.applymap(clean_text)

def fuzzy_deduplicate(df, threshold=90):
    string_cols = df.select_dtypes(include=['object', 'string']).columns
    indices_to_drop = set()
    for i in range(len(df)):
        if i in indices_to_drop:
            continue
        row_i = df.iloc[i][string_cols].astype(str).str.lower().fillna('').values
        for j in range(i + 1, len(df)):
            if j in indices_to_drop:
                continue
            row_j = df.iloc[j][string_cols].astype(str).str.lower().fillna('').values
            sim_scores = [fuzz.ratio(a, b) for a, b in zip(row_i, row_j)]
            avg_score = sum(sim_scores) / len(sim_scores)
            if avg_score >= threshold:
                indices_to_drop.add(j)
    return df.drop(index=indices_to_drop).reset_index(drop=True)

def semantic_deduplicate(df, column='sentence', threshold=0.9):
    if hub is None or faiss is None:
        raise ImportError("Both faiss and tensorflow_hub are required for semantic deduplication.")

    embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
    sentences = df[column].astype(str).tolist()
    embeddings = [embed([text]).numpy() for text in sentences]
    embeddings = np.concatenate(embeddings, axis=0)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    dedup_sentence = set()
    unique_duplicated_pairs = set()
    duplicated_indices = set()

    for i, embedding in enumerate(embeddings):
        D, I = index.search(np.array([embedding]), k=len(embeddings))
        for k, j in enumerate(I[0]):
            if j != i and D[0][k] >= threshold and j not in duplicated_indices:
                pair = (sentences[i], sentences[j])
                if pair not in unique_duplicated_pairs:
                    unique_duplicated_pairs.add(pair)
                    dedup_sentence.add(sentences[i])
                    dedup_sentence.add(sentences[j])
                    duplicated_indices.add(j)

    df['Flag'] = df[column].apply(lambda x: x in dedup_sentence)
    df_dedup = df[~df['Flag']].drop(columns=['Flag']).reset_index(drop=True)
    return df_dedup