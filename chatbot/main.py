import nltk
#nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

##################### Pre-processing ###############

# laad de json file met alle info
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
       words, labels, training, output = pickle.load(f)
except:
    # stores tokenized word from the pattern
    words = []
    labels = []
    # stores pattern
    docs_x = []
    # stores tag of the pattern
    docs_y = []

    # Loop door dictionaries voor elke intent
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Stemming - Pakt elke zin en kijkt naar de belangrijkste woord in de pattern
    # alles naar lower-case
    # sorted list zodat er geen duplicates in zitten
    # ? wordt genegeerd als het  getypt wordt
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    # training one hot encoding
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        pattern_words = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in pattern_words:
                # woord bestaat
                bag.append(1)
            else:
                # woord bestaat niet
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    # save data in pickle file om te loaden voor volgende keer
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output),f)

#######################################################
################ Train with tensforflow ###############
# reset graph
tensorflow.reset_default_graph()
# maak neural network
net = tflearn.input_data(shape=[None,len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

####################################################
############## Chat met Bot ########################

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)
    return np.array(bag)

def chat():
    # Onthoud naam in een text file. kan maar 1 naam tegelijk onthouden.
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    fi = open("name.txt","r+")
    name = fi.readline()
    if name != "":
        print("Hello " + name + " Welcome back!")
        fi.close()
    else:
        name = input("I've not seen you here before, what is your name?\nYou:    ")
        print("Hello " + str(name) +" , nice to meet you!")
        fi.write(name)
        fi.close()

    while True:
        inp = input("You:    ")
        if inp.lower() == "bye":
            break

        # geeft antwoord terug gebaseerd op het antwoord met de hoogste kans.
        results = model.predict([bag_of_words(inp,words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        # boven 70% confidence print het response van de tag en anders print het iets anders
        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print("Tim:  " + random.choice(responses))
        else:
            for tg in data["intents"]:
                if tg['tag'] == "random-nl":
                    responses = tg['responses']
            print("Tim:  " + random.choice(responses))

chat()


