# Kaggle based Data Analysis
# ham is normal, spam is spam

# Scikit-learn help with classification, regresion, fit() and predict(), score() and predict_score()

# Data Analysis
# teach set and test set
# teach is for training the model
# test is for testing the model for accuracy

from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

# Load the data
def perform(classifiers, vectorizers, train_data, test_data): # classifiers and vectorizers are lists
    for classifier in classifiers:
        for vectorizer in vectorizers:
            string = ''
            string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__ 
            # the line above adds the classifier and vectorizer to the string __class__.__name__ is the name of the class
            
            # train the model
            vectorize_text = vectorizer.fit_transform(train_data.v2) # v2 is the text and fit_transform is a method of the vectorizer
            classifier.fit(vectorize_text, train_data.v1) # fit is a method of the classifier fits orginal data

            # test the model
            vectorize_text = vectorizer.transform(test_data.v2)
            # transform is a method of the vectorizer that transforms the data
            #fit-transform is used for training the model and transform is used for testing the model
            score = classifier.score(vectorize_text, test_data.v1) # score is a method of the classifier that scores the data
            string += ' scored ' + str(score)
            print(string) # nothing is returned because the function is void

