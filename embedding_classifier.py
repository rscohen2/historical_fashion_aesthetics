import numpy as np
import pandas as pd
# import sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


#set up the model for creating embeddings
from sentence_transformers import SentenceTransformer

# def get_embeddings(titles_list):
#     embeddings = []
#     if isinstance(titles_list, str):
#         # titles_list = [titles_list]
#         titles_list.strip('[')
#         titles_list.strip(']')
#         titles_list = titles_list.split(' ')
#         for title in titles_list:
#             title = str(title)
#             title = title.strip('[')
#             title = title.strip(']')
#             title = title.strip("'")
#             embedding = model.encode([title])
#             embeddings.append(embedding)
#         return embeddings
#     else:
#         return model.encode('')

    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer('all-MiniLM-L6-v2')

    df['sentence_emb'] = df['sentence'].apply(lambda x: model.encode(x))
    df['adj_emb'] = df['adjectives'].apply(
        lambda x: model.encode(' '.join(x)) if isinstance(x, list) else model.encode(str(x)))
    df['fashion_emb'] = df['fashion_terms'].apply(
        lambda x: model.encode(' '.join(x)) if isinstance(x, list) else model.encode(str(x)))


df['combined_emb'] = df.apply(
    lambda row: np.concatenate([row['sentence_emb'], row['adj_emb'], row['fashion_emb']]),
    axis=1
)
X = np.vstack(df['combined_emb'])
y = df['label']  # or whatever your target column is



model = SentenceTransformer("all-mpnet-base-v2")

#we want to find the most similar adjective from our list to the embedding the model learns from the most

