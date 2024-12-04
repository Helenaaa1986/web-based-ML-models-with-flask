import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_data(file, options):
    nltk.download('punkt')
    # Load the Excel file into a pandas dataframe
    df = pd.read_excel(file)

    # Define the preprocessing operations based on user's selection
    lowercase = False  # set to True if user selects lowercase
    lemmatize = False  # set to True if user selects lemmatization
    remove_stopwords = False  # set to True if user selects stopword removal
    
    for option in options:
        if option == 'stopwords':
            remove_stopwords = True
        if option == 'lowercase':
            lowercase = True
        if option == 'lemmatize':
            lemmatize = True

    # Define the NLTK objects needed for preprocessing
    nltk.download("stopwords")
    nltk.download("wordnet")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Loop through each column of the dataframe and apply the preprocessing operations
    for column in df.columns:
        # Skip non-string columns
        if df[column].dtype != "object":
            continue

        # Apply the preprocessing operations to each cell in the column
        for i, cell in df[column].items():
            if pd.isna(cell):
                continue

            text = str(cell)
            if lowercase:
                text = text.lower()
            if lemmatize:
                words = nltk.word_tokenize(text)
                lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
                text = " ".join(lemmatized_words)
            if remove_stopwords:
                words = nltk.word_tokenize(text)
                filtered_words = [word for word in words if word not in stop_words]
                text = " ".join(filtered_words)

            # Update the cell value in the dataframe
            df.at[i, column] = text
        break    
    # Save the preprocessed dataframe back to the Excel file
    return df
    # excel_file =  df.to_excel("/content/DatasetEN-Categories.xlsx", index=False)
    

