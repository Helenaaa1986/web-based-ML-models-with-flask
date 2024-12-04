from sklearn import svm
import threading
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import time
import matplotlib
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE

matplotlib.use('Agg')

def save_commonplot(wordcloud, i):
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Common Words in Cluster {i+1}")
    plt.savefig(f"static/common_words_{i}.png")
    plt.close()

def save_uniqueplot(wordcloud, i):
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Unique Words in Cluster {i+1}")
    plt.savefig(f"static/unique_words_{i}.png")
    plt.close()


def naivebayes_model(df, **parameters):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    # Create and fit the vectorizer
    vectorizer = TfidfVectorizer()
    vectorize_data = vectorizer.fit_transform(text_data)
    smote_param = parameters.get("smote", False)
    if smote_param == 'True' or smote_param == True:
        smote = SMOTE(random_state=42)
        vectorize_data_resampled, labels_resampled = smote.fit_resample(vectorize_data, labels)
    else:
        vectorize_data_resampled, labels_resampled = vectorize_data, labels

    y = labels_resampled
    X_train, X_test, y_train, y_test = train_test_split(vectorize_data_resampled, labels_resampled, test_size=0.1, random_state=42)

    # Create and fit the Multinomial Naive Bayes classifier
    alpha = float(parameters.get("alpha", 1.0))
    fit_prior = parameters.get("fit_prior", True)
    kfolds = int(parameters.get("kfold", 2 if parameters.get("kfold") == '' else parameters.get("kfold")))
    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X_train, y_train)

    # Save the model and vectorizer
    joblib.dump(clf, 'naivebayes_model.pkl')
    joblib.dump(vectorizer, 'nbvectorizer.pkl')

    # Predict on the test data
    y_pred = clf.predict(X_test)
    classification = classification_report(y_test, y_pred, output_dict=True)

    scores = cross_val_score(clf, vectorize_data_resampled, y, cv=kfolds)
    accuracy = scores.mean()
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)

    # Calculate learning curve data
    train_sizes, train_scores, test_scores = learning_curve(clf, vectorize_data_resampled, y, cv=kfolds, scoring='accuracy', n_jobs=-1)
    learning_curve_data = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores
    }

    return accuracy, recall, precision, cm, learning_curve_data

def decisiontree_model(df, **parameters):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    # Vectorize your data
    vectorizer = CountVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)
    
    smote_param = parameters.get("smote", False)
    if smote_param == 'True' or smote_param == True:
        smote = SMOTE(random_state=42)
        vectorized_data_resampled, labels_resampled = smote.fit_resample(vectorized_data, labels)
    else:
        vectorized_data_resampled, labels_resampled = vectorized_data, labels

    X_train, X_test, y_train, y_test = train_test_split(vectorized_data_resampled, labels_resampled, test_size=0.1, random_state=42)

    # Convert max_depth to an integer
    max_depth = int(parameters.get("max_depth", 2 if parameters.get("max_depth") == '' else parameters.get("max_depth")))

    # Train your decision tree model
    clf = DecisionTreeClassifier(criterion=parameters.get("criterion", "gini"), max_depth=max_depth)
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'decisiontree_model.pkl')
    vectorizer_filename = 'dtvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    kfolds = int(parameters.get("kfold", 2 if parameters.get("kfold") == '' else parameters.get("kfold")))
    scores = cross_val_score(clf, vectorized_data_resampled, labels_resampled, cv=kfolds, scoring='accuracy')

    # Evaluate your decision tree model
    y_pred = clf.predict(X_test)
    accuracy = scores.mean()
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Learning curve data
    train_sizes, train_scores, test_scores = learning_curve(clf, vectorized_data_resampled, labels_resampled, cv=kfolds, scoring='accuracy', n_jobs=-1)
    learning_curve_data = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores
    }

    return accuracy, recall, precision, cm, learning_curve_data

def svm_model(df, **parameters):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)

    smote_param = parameters.get("smote", False)
    if smote_param == 'True' or smote_param == True:
        smote = SMOTE(random_state=42)
        vectorized_data_resampled, labels_resampled = smote.fit_resample(vectorized_data, labels)
    else:
        vectorized_data_resampled, labels_resampled = vectorized_data, labels


    # Split resampled data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(vectorized_data_resampled, labels_resampled, test_size=0.1, random_state=42)

    # Set SVM parameters
    kernel = parameters.get("kernel", "linear")
    C = float(parameters.get("C", 1.0))
    degree = int(parameters.get("degree", 3))

    # Use a ternary operator to set kfolds with a default value of 5 if not provided or if it's an empty string
    kfolds = int(parameters.get("kfold", 2 if parameters.get("kfold") == '' else parameters.get("kfold")))

    # Train SVM classifier
    clf = svm.SVC(kernel=kernel, C=C, degree=degree)
    clf.fit(X_train, y_train)

    joblib.dump(clf, 'svm_model.pkl')
    vectorizer_filename = 'svmvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Predict test set
    y_pred = clf.predict(X_test)

    classification = classification_report(y_test, y_pred, output_dict=True)

    # Cross-validate with SMOTE-resampled data
    scores = cross_val_score(clf, vectorized_data_resampled, labels_resampled, cv=kfolds)

    accuracy = scores.mean()
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)

    # Learning curve data
    train_sizes, train_scores, test_scores = learning_curve(clf, vectorized_data_resampled, labels_resampled, cv=kfolds, scoring='accuracy', n_jobs=-1)
    learning_curve_data = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores
    }

    return accuracy, recall, precision, cm, learning_curve_data

def knn_model(df, **parameters):
    text_data = df.iloc[:, 0]
    labels = df.iloc[:, 1]

    n_neighbors = int(parameters.get("n_neighbors", 3))
    kfolds = int(parameters.get("kfold", 3))

    # Vectorize text data
    vectorizer = TfidfVectorizer()
    vectorized_data = vectorizer.fit_transform(text_data)

    smote_param = parameters.get("smote", False)
    if smote_param == 'True' or smote_param == True:
        smote = SMOTE(random_state=42)
        vectorized_data_resampled, labels_resampled = smote.fit_resample(vectorized_data, labels)
    else:
        vectorized_data_resampled, labels_resampled = vectorized_data, labels

    # Split the resampled dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(vectorized_data_resampled, labels_resampled, test_size=0.1, random_state=42)

    # Create a KNN classifier object with n_neighbors
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Train the classifier on the training set
    knn.fit(X_train, y_train)

    joblib.dump(knn, 'knn_model.pkl')
    vectorizer_filename = 'knnvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Perform k-fold cross-validation on the resampled data
    scores = cross_val_score(knn, vectorized_data_resampled, labels_resampled, cv=kfolds)

    # Calculate mean accuracy over k-folds
    accuracy = scores.mean()

    # Make predictions on the testing set
    y_pred = knn.predict(X_test)

    classification = classification_report(y_test, y_pred, output_dict=True)
    recall = classification['macro avg']['recall']
    precision = classification['macro avg']['precision']
    cm = confusion_matrix(y_test, y_pred)

    # Learning curve data
    train_sizes, train_scores, test_scores = learning_curve(knn, vectorized_data_resampled, labels_resampled, cv=kfolds, scoring='accuracy', n_jobs=-1)
    learning_curve_data = {
        "train_sizes": train_sizes,
        "train_scores": train_scores,
        "test_scores": test_scores
    }

    return accuracy, recall, precision, cm, learning_curve_data

def kmean_model(df, **parameters):
    # Convert text data to TF-IDF vectors
    text_data = df.iloc[:, 0]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)

    # Apply SMOTE to balance the dataset if specified
    smote_param = parameters.get("smote", False)
    if smote_param == 'True' or smote_param == True:
        smote = SMOTE(random_state=0)
        X_resampled, labels_resampled = smote.fit_resample(X, df.iloc[:, 1])  # Assuming df.iloc[:, 1] contains your labels
    else:
        X_resampled, labels_resampled = X, df.iloc[:, 1]

    # Convert hyperparameters to integers
    n_clusters = int(parameters.get("n_clusters", 3))
    n_init = int(parameters.get("n_init", 50))
    max_iter = int(parameters.get("max_iter", 300))
    init = parameters.get("init", "k-means++")

    # Perform K-Means clustering with specified hyperparameters
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, init=init, random_state=0)
    labels = kmeans.fit_predict(X_resampled)

    joblib.dump(kmeans, 'kmean_model.pkl')
    vectorizer_filename = 'kmeanvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Get common words and unique words in each cluster
    common_words = []
    unique_words = []
    for i in range(n_clusters):
        cluster_words = []
        for j in range(len(text_data)):
            if labels[j] == i:
                cluster_words += text_data[j].split()
        common_words.append(set(cluster_words))
        unique_cluster_words = set(cluster_words) - set().union(*common_words)
        unique_words.append(unique_cluster_words)
        common_words[-1] = common_words[-1].union(unique_cluster_words)

    silhouette_scoree = silhouette_score(X_resampled, labels)

    # Calculate confusion matrix (Silhouette score used for illustration, replace with your actual evaluation metric)
    cm = silhouette_score(X_resampled, labels)

    # Generate word clouds for each cluster (you need to implement this function)
    for i in range(n_clusters):
        if common_words[i]:
            wordcloud = WordCloud(background_color="white").generate(' '.join(common_words[i]))
            save_commonplot(wordcloud, i)
        if unique_words[i]:
            wordcloud = WordCloud(background_color="white").generate(' '.join(unique_words[i]))
            save_uniqueplot(wordcloud, i)

    return text_data, labels_resampled, silhouette_scoree

def dbscan_model(df, **parameters):
    data = df.iloc[:, 0]

    # Vectorize data
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data)

    smote = SMOTE(random_state=0)
    X_resampled, labels_resampled = smote.fit_resample(X, df.iloc[:, 1])  # Assuming df.iloc[:, 1] contains your labels

    # Convert hyperparameters to appropriate types
    eps = float(parameters.get("eps", 1.0))
    min_samples = int(parameters.get("min_samples", 50))
    metric = parameters.get("metric", "euclidean")

    # Cluster data using DBSCAN with specified hyperparameters
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(X)

    joblib.dump(dbscan, 'dbscan_model.pkl')
    vectorizer_filename = 'dbscanvectorizer.pkl'
    joblib.dump(vectorizer, vectorizer_filename)

    # Get cluster assignments
    labels = dbscan.labels_

    # Get silhouette score
    silhouette_avg = silhouette_score(X, labels)

    # Get word clouds for each cluster
    wordclouds = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_data = [data[i] for i, l in enumerate(labels) if l == label]
        cluster_X = vectorizer.transform(cluster_data)
        wordcloud = WordCloud(background_color="white").generate(" ".join(cluster_data))
        wordclouds.append((label, wordcloud.to_svg()))

    # Get common and unique words for each cluster
    common_words = {}
    unique_words = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_data = [data[i] for i, l in enumerate(labels) if l == label]
        cluster_X = vectorizer.transform(cluster_data)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sum = cluster_X.sum(axis=0)
        tfidf_scores = [(feature_names[i], tfidf_sum[0, i]) for i in range(len(feature_names))]
        tfidf_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        common_words[label] = tfidf_scores[:5]
        unique_words[label] = [w for w, s in tfidf_scores[-5:]]

    # Assign labels to clusters
    cluster_labels = {}
    for label in set(labels):
        if label == -1:
            cluster_labels[label] = "Noise"
        else:
            cluster_labels[label] = f"Cluster {label}"

    return labels, silhouette_avg, wordclouds, common_words, unique_words, cluster_labels

df = pd.read_excel("train data.xlsx")

#dbscan_model(df)

