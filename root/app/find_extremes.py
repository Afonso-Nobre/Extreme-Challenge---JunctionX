"""
This file contains code to

    (a) Load the pre-trained classifier and
    associated files.

    (b) Transform new input data into the
    correct format for the classifier.

    (c) Run the classifier on the transformed
    data and return results.
"""

import pickle
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.stem.porter import *
import string
import re
from scipy.sparse import hstack

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *


stopwords=nltk.corpus.stopwords.words("english") # content words only for higher efficiency

sentiment_analyzer = VS()

stemmer = PorterStemmer() # to get just the stem of the words

def tokenize(sentence):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", sentence.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def get_pos_tags(sentences):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS = Part of Sentence tags).
    """
    tweet_tags = []
    for s in sentences:
        tokens = tokenize(s)
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags

def features(sentence):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features.

    This is modified to only include those features in the final
    model."""

    syllables = textstat.syllable_count(sentence) #count syllables in words
    num_chars = sum(len(w) for w in sentence) #num chars in words
    num_chars_total = len(sentence)
    num_words = len(sentence.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(sentence.split()))

    # Below are scores for readability of text:

    # Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)

    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_words, num_unique_terms]
    #features = pandas.DataFrame(features)
    return features

def get_oth_features(sentences):
    """Takes a list of tweets, generates features for
    each tweet, and returns a numpy array of tweet x features"""
    feats=[]
    for s in sentences:
        feats.append(features(s))
    return np.array(feats)

def predictions(matrix, model):
    """
    This function calls the predict function on
    the trained model to generated a predicted y
    value for each observation.
    """
    y_preds = model.predict(matrix)
    return y_preds

def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    if class_label == 0:
        return "hate speech"
    elif class_label == 1:
        return "offensive language"
    elif class_label == 2:
        return "appropriate language"
    else:
        return "no idea"

def get_sentences_predictions(sentences, data, labels):
    print(f"{len(sentences)} sentences to classify")

    # Train vectorizers
    tf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    matrix_tf = tf_vectorizer.fit_transform(data)

    pos_vectorizer = CountVectorizer()
    matrix_pos = pos_vectorizer.fit_transform(get_pos_tags(data))

    # Combine TF and POS
    matrix_combined = hstack([matrix_tf, matrix_pos])

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(matrix_combined, labels)

    # Transform input sentences with SAME vectorizers
    matrix_tf_new = tf_vectorizer.transform(sentences)
    matrix_pos_new = pos_vectorizer.transform(get_pos_tags(sentences))
    matrix_new = hstack([matrix_tf_new, matrix_pos_new])

    # Predict
    predicted_class = predictions(matrix_new, clf)
    return predicted_class


def find_extremes(sentences) :

    # Example training data
    data = [
        "People from different backgrounds bring value to society",  # neither
        "Immigrants from certain groups are ruining the country",  # hate
        "Everyone deserves equal treatment regardless of race",  # neither
        "Some races are inferior to others",  # hate
        "That person is so dumb",  # offensive
    ]
    labels = [2, 0, 2, 0, 1]  # 0=hate, 1=offensive, 2=neither

    # New sentences to classify
    """sentences = [
        "Racism should be eliminated from society",
        "People from that group are dangerous",
        "You idiot, stop talking"
    ]"""
    sentences_predictions = get_sentences_predictions(sentences, data, labels)

    print("Printing predicted values: ")
    result = ""
    for i,s in enumerate(sentences):
        print(i, sentences[i])
        print("Classified as " + class_to_name(sentences_predictions[i]))
        result += str(i) + " is: " + str(sentences[i]) + "\n" + "And is classified as: " + str(class_to_name(sentences_predictions[i]))

    # print("Calculate accuracy on labeled data")
    # sentences = []
    # labels = []
    # predictions = get_sentences_predictions(sentences)
    # right_count = 0
    # for i,s in enumerate(sentences):
        #     if sentences_class[i] == predictions[i]:
    #         right_count += 1

    # accuracy = right_count / float(len(df))
    # print "accuracy", accuracy

    return result