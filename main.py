import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import tensorflow
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

print(data)
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
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
            docs_x.append(token_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # get stem word, "root of the word" because its the only thing we care about
    words = [stemmer.stem(w.lower())for w in words if w not in "q?"] # get rid of question mark
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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

#input = bag of words, out put is tags
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        print(results)
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))

chat()