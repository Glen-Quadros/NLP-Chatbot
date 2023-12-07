import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
  global_settings = {}
  global_settings.update(configuration)
  print("Configuration updated:", global_settings)

f = open('/content/data.txt', 'r', errors='ignore')
raw_doc = f.read()

raw_doc = raw_doc.lower()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
sentence_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]

remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict)))

greet_inputs = ('hello', 'hi', 'whassup', 'how are you')
greet_responses = ('hi', 'Hey', 'Hey There!', 'There there!!')

def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)

def response(user_response):
  robo1_response = ''
  TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
  tfidf = TfidfVec.fit_transform(sentence_tokens)
  vals = cosine_similarity(tfidf[-1], tfidf)
  idx = vals.argsort()[0][-2]
  flat = vals.flatten()
  flat.sort()
  req_tfidf = flat[-2]

  if req_tfidf == 0:
    robo1_response = robo1_response + 'I am sorry. I am unable to process your request'
  else:
    robo1_response = robo1_response + sentence_tokens[idx]
    return robo1_response

############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # Preprocess the text
        sentence = text.lower()
        sentence = nltk.word_tokenize(sentence)

        # Check for greeting
        response = greet(sentence)
        if response:
            output.append(response)
            continue

        # Vectorize and find similar sentence
        TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
        tfidf = TfidfVec.fit_transform(sentence_tokens)
        vals = cosine_similarity(tfidf[-1], tfidf)
        idx = vals.argsort()[0][-2]
        flat = vals.flatten()
        flat.sort()
        req_tfidf = flat[-2]

        # Prepare response
        if req_tfidf == 0:
            response = 'I am sorry. I am unable to process your request'
        else:
            response = sentence_tokens[idx]

        # Append response and continue
        output.append(response)

    return SimpleText(dict(text=output))


if __name__ == '__main__':
    openfabric = OpenfabricExecutionRay()
    openfabric.set_callback_config(config)
    openfabric.set_callback_execute(execute)
    openfabric.run()