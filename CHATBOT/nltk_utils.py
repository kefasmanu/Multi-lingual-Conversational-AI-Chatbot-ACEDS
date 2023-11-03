import numpy as np
import nltk
# nltk.download('punkt')
import spacy
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)



nlp = spacy.load('en_core_web_sm')

def lemmatize_text(word):
    """
    lemmatize = find the root form of the word
    It takes a text as input and returns the lemmatized version of the text. 
    It process the text using the loaded spaCy model and iterate over the 
    tokens to extract their lemmas using the token.lemma_ attribute. 
    Finally, it join the lemmas back into a string and return the lemmatized text.
    
    """
    doc = nlp(word)
    lemmas = [token.lemma_.lower() for token in doc]
    return ' '.join(lemmas)


# def stem(word):
#     """
#     stemming = find the root form of the word
#     examples:
#     words = ["organize", "organizes", "organizing"]
#     words = [stem(w) for w in words]
#     -> ["organ", "organ", "organ"]
#     """
#     return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # lemmatize each word
    sentence_words = [lemmatize_text(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
