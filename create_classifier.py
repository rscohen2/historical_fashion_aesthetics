import numpy as np
import pandas as pd
import sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


#set up the model for creating embeddings
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer("all-mpnet-base-v2")
###

# df = pd.read_csv('fashion_mentions.csv', dtype={'adjectives': 'string'})
df = pd.read_csv('fashion_mentions.csv')



def get_embeddings(titles_list):
    embeddings = []
    if isinstance(titles_list, str):
        # titles_list = [titles_list]
        titles_list.strip('[')
        titles_list.strip(']')
        titles_list = titles_list.split(' ')
        for title in titles_list:
            title = str(title)
            title = title.strip('[')
            title = title.strip(']')
            title = title.strip("'")
            embedding = model.encode([title])
            embeddings.append(embedding)
        return embeddings
    else:
        return model.encode('')




# X = df.drop(['gender','character_start_idx','character_end_idx','book_id','adjectives','adj_embeddings'],axis=1)
print(df.columns)

df['gender'] = df['gender'].replace({'she/her': 1, 'he/him/his': 0, 'they/them/their':2})

df = df.dropna()

df['adj_embeddings'] = df['adjectives'].apply(get_embeddings)

df['avg_adj_embeddings'] = df['adj_embeddings'].apply(lambda embs: np.mean(embs, axis=0))

Y = df['gender']


X = np.vstack(df['avg_adj_embeddings'].values)


# print(df['adj_embeddings'].dtype)

import requests
# from huggingface_hub import configure_http_backend
#
# def backend_factory() -> requests.Session:
#     session = requests.Session()
#     session.verify = False
#     return session
#
# configure_http_backend(backend_factory=backend_factory)

#
# print(df[df.isnull().any(axis=1)])
# print(df.isnull().values.any())
#
# len()
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

#
# #
# #Create classifier
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_1 = RandomForestRegressor(n_estimators=5)
# regr_1.fit(X,Y)
clf = RandomForestClassifier(n_estimators=5)
Y_pred=clf.predict(X)

r2_score(Y, Y_pred)

print('Accuracy: ',r2_score(Y, Y_pred))

importances = list(clf.feature_importances_)
#Print out the feature and importances
print (importances)

confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
print(sns.heatmap(confusion_matrix, annot=True))