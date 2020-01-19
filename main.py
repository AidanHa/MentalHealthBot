import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import tensorflow
import json

with open("intents.json") as file:
    data = json.load(file)

print(data)

#all words in all of our patterns
words = []
#all tags and sections
labels = []
#list of all of the patterns
docs_x = []
#tags for each corresponding pattern
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:

        token_words = nltk.word_tokenize(pattern) #returns a list with all the tokenized words (whats goes to what, etc)
        words.extend(token_words) #extend list and add all th words in words[]
        docs_x.append(pattern)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# get stem word, "root of the word" because its the only thing we care about
words = [stemmer.stem(w.lower())for w in words]
words = sorted(list(set(words)))

labels = sorted(labels)

#THIS IS HOW INPUT IS FORMED
#bag of words forany given pattern, one hot encoding
#[0,1,0,1,1,1,0,0,0,0] word 2 exist once, word 4 exist once, word 5 exist once in our sentence
#so our pattern sentences are just lists of numbers

training = []
output = []

#OUTPUT IS BASICALLY THE SAME ONE HOT ENCODER BUT FOR THE RESPONSE
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    token_words = [stemmer.stem(w) for w in doc]

    for w in words:
        if w in token_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)
