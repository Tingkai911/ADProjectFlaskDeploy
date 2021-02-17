import sqlalchemy
import pyodbc

import nltk
import numpy as np
import string
import pandas as pd
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from urllib import parse

# User environment variables for database connection
from dotenv import load_dotenv
import os
load_dotenv()
ConnectionString = os.getenv("CONNECTION_STRING")

def main():
    corpus = getDataFromDB()

    stop_words = stopwords.words('english')

    porter = nltk.stem.PorterStemmer()

    docs = []
    punc = str.maketrans('', '', string.punctuation)
    for doc in corpus:
        doc_no_punc = doc.translate(punc)
        words_stemmed = [porter.stem(w) for w in doc_no_punc.lower().split()
                        if not w in stop_words]
        docs += [' '.join(words_stemmed)]

    print(docs)

    tfidf_vec = TfidfVectorizer()
    tfidf_wm = tfidf_vec.fit_transform(docs).toarray()
    features = tfidf_vec.get_feature_names()
    tfidf_df = pd.DataFrame(data=tfidf_wm, columns=features)
    print(tfidf_df)

    # save pre calculated tf-idf values
    filename = "allergen_dataframe.csv"
    tfidf_df.to_csv(filename, index=False)

    # save the model
    filename = 'allergen_model.sav'
    joblib.dump(tfidf_vec, filename)

    print("Complete")

def getDataFromDB():
    params = parse.quote_plus(ConnectionString)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    tag_name = []

    with engine.connect() as con:
        rs = con.execute('SELECT TagId, tagName FROM Tag ORDER BY TagId')
        for row in rs:
            tag_name.append(row[1])
            
    return tag_name

if __name__ == "__main__":
    main()