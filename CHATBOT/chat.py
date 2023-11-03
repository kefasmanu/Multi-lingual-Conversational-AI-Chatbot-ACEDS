# import random
# import json

# import torch

# from model import NeuralNet
# from nltk_utils import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# with open('intents.json', 'r') as json_data:
#     intents = json.load(json_data)

# FILE = "data.pth"
# data = torch.load(FILE)
# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data['all_words']
# tags = data['tags']
# model_state = data["model_state"]

# model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state)
# model.eval()

# bot_name = "Mimi"

# def get_response(msg):
#     sentence = tokenize(msg)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)

#     output = model(X)
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]
#     if prob.item() > 0.75:
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 return random.choice(intent['responses'])
    
#     return "I dont have idea, you can be more precise about what you want. What else can I help you with?"


# if __name__ == "__main__":
#     print("Let's chat! (type 'quit' to exit)")
#     while True:
#         # sentence = "do you use credit cards?"
#         sentence = input("You: ")
#         if sentence == "quit":
#             break

#         resp = get_response(sentence)
#         print(resp)

import random
import json
from langdetect import detect
import torch
import numpy as np

import requests
import json
from translator import translator

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = ""
#print("Let's chat! (type 'quit' to exit)")

def query(question : str):
  # sentence = "How may i help you?"
  sentence = question

  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  output = model(X)
  _, predicted = torch.max(output, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]

  # Print accuracy and confidence to the log
  accuracy = np.max(probs.cpu().detach().numpy())
  confidence = prob.item()
  print(f"Accuracy: {accuracy}, Confidence: {confidence}")

  if prob.item() > 0.75:
      for intent in intents['intents']:
          if tag == intent["tag"]:
              return f"{bot_name}{random.choice(intent['responses'])}"
  else:
      return f"{bot_name}I dont get your query well. Any further request?"






# class translator:
#     api_url = "https://translate.googleapis.com/translate_a/single"
#     client = "?client=gtx&dt=t"
#     dt = "&dt=t"

#     #fROM English to Kinyarwanda
#     def translate(text : str , target_lang : str, source_lang : str):
#         sl = f"&sl={source_lang}"
#         tl = f"&tl={target_lang}"
#         r = requests.get(translator.api_url+ translator.client + translator.dt + sl + tl + "&q=" + text)
#         return json.loads(r.text)[0][0][0]


#processing text and language
def process_question(text : str, lang):

  #source_lang = detect(text)
  resp = translator.translate(text=text, target_lang="en", source_lang=lang)
  return resp


def process_answer(text : str, source_lang):
  resp = translator.translate(text=text, target_lang=source_lang, source_lang='en')
  return resp

#Message and language
def process_message(QUESTION: str, lang: str):
  USER_QUERY = process_question(QUESTION, lang) #Translate the original question into english and store the source lang
  RESPONSE = query(USER_QUERY) #Asking th chatbot question
  ORIGINAL_RESPONSE = process_answer(RESPONSE, lang)
  return ORIGINAL_RESPONSE

#print(process_message("Information about ACEDS?"))