from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
from nltk.stem.porter import PorterStemmer
import re
from scipy.sparse import hstack, csr_matrix
from textstat.textstat import textstat

database_dir = Path.cwd().parent / "data" / "database"

# Get all CSV files in the folder
csv_files = list(database_dir.glob("*.csv"))

dfs = [pd.read_csv(f, usecols=["sentence", "label"]) for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

data = df["sentence"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

stemmer = PorterStemmer()

def tokenize(sentence):
    tweet = " ".join(re.split("[^a-zA-Z]*", sentence.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def get_pos_tags(sentences):
    pos_list = []
    for s in sentences:
        tokens = tokenize(s)
        tags = nltk.pos_tag(tokens)
        tag_str = " ".join(tag for _, tag in tags)
        pos_list.append(tag_str)
    return pos_list


def features(sentence):
    syllables = textstat.syllable_count(sentence)
    num_chars_total = len(sentence)
    num_words = len(sentence.split())
    avg_syl = round((syllables + 0.001) / (num_words + 0.001), 4)
    num_unique_terms = len(set(sentence.split()))

    FKRA = round(0.39 * num_words + 11.8 * avg_syl - 15.59, 1)
    FRE = round(206.835 - 1.015 * num_words - 84.6 * avg_syl, 2)

    return [FKRA, FRE, syllables, num_chars_total, num_words, num_unique_terms]

def get_oth_features(sentences):
    feats = []
    for s in sentences:
        feats.append(features(s))
    return np.array(feats)

tf_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=5000)
matrix_tf = tf_vectorizer.fit_transform(data)

pos_vectorizer = CountVectorizer()
matrix_pos = pos_vectorizer.fit_transform(get_pos_tags(data))

matrix_other = csr_matrix(get_oth_features(data))  # convert to sparse

# Combine all feature sets
matrix_combined = hstack([matrix_tf, matrix_pos, matrix_other])

clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
clf.fit(matrix_combined, labels)

def class_to_name(class_label):
    if class_label == 0:
        return "hate speech"
    elif class_label == 1:
        return "offensive language"
    elif class_label == 2:
        return "appropriate"
    else:
        return "no idea"

def predict_sentences(sentences):
    matrix_tf_new = tf_vectorizer.transform(sentences)
    matrix_pos_new = pos_vectorizer.transform(get_pos_tags(sentences))
    matrix_other_new = csr_matrix(get_oth_features(sentences))
    matrix_new = hstack([matrix_tf_new, matrix_pos_new, matrix_other_new])
    preds = clf.predict(matrix_new)
    return preds

parent_dir = Path.cwd().parent
en_path = parent_dir/"data"/"database"/"en.txt"
# reads the database
with open(en_path, "r", encoding="utf-8") as f:
    bad_words = set(w.strip().lower() for w in f if w.strip())

def find_extremes(sentences, timestamps):

    if len(sentences) == 0:
        return "Empty file"

    preds = predict_sentences(sentences)

    if len(sentences) == 1:
        if preds[0] == 0 or preds[0] == 1:
            return "Short file with problems!"

    result = ""
    for i, s in enumerate(sentences):
        # offensive language
        tokens = tokenize(s)
        found = [t for t in tokens if t in bad_words]
        if found:
            result += "offensive language: "
            if i == len(sentences) - 1:
                result += str(int(timestamps[i])) + "s."
            else:
                result += str(int(timestamps[i])) + "s; \n"

        # results from the model
        if preds[i] == 0 or preds[i] == 1:
            result += "extremist view: "
            if i == len(sentences) - 1:
                result += str(int(timestamps[i])) + "s."
            else:
                result += str(int(timestamps[i])) + "s; \n"

    if result == "":
        return "No problems found"

    return "Problems found around the following times: \n" + result
