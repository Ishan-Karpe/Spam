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
import csv

data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items

classifer = OneVsRestClassifier(SVC(kernel='linear')) # a linear support vector classifier
vectorizer = TfidfVectorizer() # a count vectorizer tfdf vectorizer is a vectorizer that converts text to a matrix of tfidf features

vectorize_text = vectorizer.fit_transform(train_data.v2) # fit_transform is a method of the vectorizer that fits the data
classifer.fit(vectorize_text, train_data.v1) # fit is a method of the classifier that fits the data

csv_array = []
for index, row in test_data.iterrows(): # iterrows is a method of the dataframe that iterates over the rows
    answer = row[0]
    text = row[1]
    vectorize_text = vectorizer.transform([text])
    prediction = classifer.predict(vectorize_text)[0]
    if prediction == answer: # if the prediction is correct
        result = 'correct'
    else:
        result = 'incorrect'
    csv_array.append([text, answer, prediction, result])

with open('test_score.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', # writer is a method of the csv module that writes the data to a file
            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['#', 'text', 'answer', 'predict', result]) # write the header

    for row in csv_array:
        spamwriter.writerow(row)

        # according to my results, the model is 98% accurate with 15 wrong results and 1176 correct results
        # the model is quite accurate, now we can write to API FLASK