import numpy as np
import random
import json
import matplotlib.pyplot as plt
import seaborn as sn
sn.set()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report
from nltk_utils import bag_of_words, tokenize, lemmatize_text
from model import NeuralNet

with open('test_intent.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [lemmatize_text(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Load test data
test_xy = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        test_xy.append((w, tag))

X_test = []
y_test = []
for (pattern_sentence, tag) in test_xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_test.append(bag)
    label = tags.index(tag)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Load the trained model
FILE = "data.pth"
data = torch.load(FILE)

model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]


model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Convert test data to PyTorch tensor
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Get predictions for test data
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted_labels = torch.max(outputs, dim=1)

predicted_labels = predicted_labels.numpy()
y_test = y_test_tensor.numpy()

# Calculate classification report
classification_rep = classification_report(y_test, predicted_labels, target_names=tags)
print("Classification Report:\n", classification_rep)