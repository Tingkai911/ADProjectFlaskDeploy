"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import nltk
#nltk.download('stopwords')
import numpy as np
import string
import pandas as pd
import math
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from flask import Flask, request, jsonify
import json
import sqlalchemy
import pyodbc

from urllib import parse

# User environment variables for database connection
from dotenv import load_dotenv
import os
load_dotenv()
ConnectionString = os.getenv("CONNECTION_STRING")

application = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
# wsgi_app = application.wsgi_app

@application.route('/')
def hello():
    """Renders a sample page."""
    getDataFromDB()
    return "Hello World!"

@application.route("/api/receiveData", methods=["POST"])
def receiveData():
    try:
        data = request.get_json()     
        corpus_recipe = data["firstName"]
        print(corpus_recipe)
        response = jsonify(data)
        return response
        #print(response["UserId"])
    except:
        exception_message = "failedlel"
        response = json.dumps({"content": exception_message})
        return response

@application.route("/api/allergentags", methods=["POST"])
def generate_allergen_tage():
    data = request.get_json()
    corpus_recipe = data["ingredients"]

    print(corpus_recipe)

    # Load the csv, will read from database here in the future
    # df = pd.read_csv("FoodData.csv")
    # corups_allergen = df["Allergy"].tolist()
    # corpus_ingredient = df["Food"].tolist()
    corpus_ingredient, corups_allergen = getDataFromDB()

    # Load the pre-calculated TF-IDF for the allergen tags
    tfidf_allergen = pd.read_csv("allergen_dataframe.csv")

    # Load the NLP Model
    loaded_model = joblib.load("allergen_model.sav")

    # Calculate the TF-IDF for the recipe
    stop_words = stopwords.words('english')
    porter = nltk.stem.PorterStemmer()
    docs = []
    punc = str.maketrans('', '', string.punctuation)
    for doc in corpus_recipe:
        doc_no_punc = doc.translate(punc)
        words_stemmed = [porter.stem(w) for w in doc_no_punc.lower().split()
                        if not w in stop_words]
        docs += [' '.join(words_stemmed)]
    print(docs)
    tfidf_wm = loaded_model.transform(docs).toarray()
    features = loaded_model.get_feature_names()
    tfidf_recipes = pd.DataFrame(data=tfidf_wm, columns=features)
    print(tfidf_recipes)

    # Calculate cosine similarity
    docs_similarity = cosine_similarity(tfidf_recipes, tfidf_allergen)
    query_similarity = docs_similarity[0]
    query_similarity

    series = pd.Series(query_similarity, index=tfidf_allergen.index)
    sorted_series = series.sort_values(ascending=False)
    sorted_series = sorted_series[sorted_series != 0]
    print(sorted_series)

    # Get the Tags
    tags = []
    for index in sorted_series.index:
        print("[Ingredient = ", corpus_ingredient[index], "]\n", corups_allergen[index], " [score = ", sorted_series[index], "]\n", sep='')
        tags.append(corups_allergen[index])
    print(tags)

    return_data = { "allergens": tags }

    return jsonify(return_data)

def getDataFromDB():
    params = parse.quote_plus(ConnectionString)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    tag_name = []
    warning = []

    with engine.connect() as con:
        rs = con.execute('SELECT TagId, tagName, warning FROM Tag ORDER BY TagId')
        for row in rs:
            tag_name.append(row[1])
            warning.append(row[2])
            
    return tag_name, warning


if __name__ == '__main__':
    application.run(port=5000, debug=True)
    # HOST = os.environ.get('SERVER_HOST', 'localhost')
    # try:
    #     PORT = int(os.environ.get('SERVER_PORT', '5555'))
    # except ValueError:
    #     PORT = 5555
    # application.run(HOST, 5000, debug=True)
