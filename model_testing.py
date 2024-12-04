import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def naivebayes_test(df):
    filename = 'naivebayes_model.pkl'
    nb_model = joblib.load(filename)

    test_data = df.iloc[:, 0]

    vectorizer = joblib.load("nbvectorizer.pkl")
    # Vectorize text data
    vectorized_data = vectorizer.transform(test_data)

    y_pred = nb_model.predict(vectorized_data)
    df = df.drop("Classes", axis=1)
    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df

def knn_test(df):
    filename = 'knn_model.pkl'
    knn_model = joblib.load(filename)

    test_data = df.iloc[:, 0]
    
    # Vectorize text data
    vectorizer = joblib.load("knnvectorizer.pkl")
    vectorized_data = vectorizer.transform(test_data)

    y_pred = knn_model.predict(vectorized_data)
    df = df.drop("Classes", axis=1)
    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df

def decisiontree_test(df):
    filename = 'decisiontree_model.pkl'
    decisiontree_model = joblib.load(filename)

    test_data = df.iloc[:, 0]

    # Vectorize text data
    vectorizer = joblib.load("dtvectorizer.pkl")
    vectorized_data = vectorizer.transform(test_data)
    
    y_pred = decisiontree_model.predict(vectorized_data)
    df = df.drop("Classes", axis=1)

    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df

def svm_test(df):
    filename = 'svm_model.pkl'
    svm_model = joblib.load(filename)

    test_data = df.iloc[:, 0]
    
    # Vectorize text data
    vectorizer = joblib.load("svmvectorizer.pkl")
    vectorized_data = vectorizer.transform(test_data)

    y_pred = svm_model.predict(vectorized_data)
    df = df.drop("Classes", axis=1)
    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df

def kmean_test(df):
    filename = 'kmean_model.pkl'
    kmean_model = joblib.load(filename)

    test_data = df.iloc[:, 0]
    
    # Vectorize text data
    vectorizer = joblib.load("kmeanvectorizer.pkl")
    vectorized_data = vectorizer.transform(test_data)

    y_pred = kmean_model.predict(vectorized_data)
    df = df.drop("Classes", axis=1)
    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df

def dbscan_test(df):
    filename = 'dbscan_model.pkl'
    dbscan_model = joblib.load(filename)

    test_data = df.iloc[:, 0]
    
    # Vectorize text data
    vectorizer = joblib.load("dbscanvectorizer.pkl")
    vectorized_data = vectorizer.transform(test_data)

    y_pred = dbscan_model.fit_predict(vectorized_data)
    df = df.drop("Classes", axis=1)

    for pred_class in y_pred:
        if pred_class == -1:
            pred_class = "Noise" 
            
    # Add predicted class labels to the input dataframe
    df['Class'] = y_pred

    return df