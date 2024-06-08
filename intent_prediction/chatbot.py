import random
import json
import pickle
import re
import numpy as np
#from intent_prediction.Typos import correct_sentence
import nltk
from keras.src.saving import load_model
from nltk.stem import WordNetLemmatizer
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from entities import extract_locations, extract_dates, extract_people_names
sid = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'D:\University\PythonProject1\intent_prediction\intents.json').read())
words = pickle.load(open(r'D:\University\PythonProject1\intent_prediction\words.pkl', 'rb'))
classes = pickle.load(open(r'D:\University\PythonProject1\intent_prediction\classes.pkl', 'rb'))
model = load_model(r'D:\University\PythonProject1\intent_prediction\chatbot_model.h5')  # The output will be numerical data


## ENTITIES
import spacy

# Load SpaCy's English model
nlp = spacy.load("en_core_web_md")

def extract_locations(text):
    """Extract locations from text."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
    return locations

def extract_people_names(text):
    """Extract people names from text."""
    doc = nlp(text)
    people_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return people_names

def extract_dates(text):
    """Extract dates from text."""
    doc = nlp(text)
    time_related_tokens = []
    for token in doc:
        if token.ent_type_ == "DATE":
            time_related_tokens.append(token.text)
    return time_related_tokens





## TYPOS


from symspellpy import SymSpell
import Levenshtein as lev
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = r"/intent_prediction\frequency_dictionary_en_82_765.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_sentence(input_term):
    terms = input_term.lower().split()
    corrected_terms = []
    for term in terms:
        if term.isalpha():
            corrected_term = sym_spell.lookup(term, max_edit_distance=2, verbosity=1)
            corrected_terms.append(corrected_term[0].term if corrected_term else term)
        else:
            corrected_terms.append(term)
    corrected_sentence = ' '.join(corrected_terms)
    return corrected_sentence, lev.distance(input_term.lower(), corrected_sentence)

#Clean up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#Converts the sentences into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, tags):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.6
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    filtered_results = [[i, r] for i, r in results if classes[i] in tags]
    print(filtered_results)
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in filtered_results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    print(return_list)
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

### for question number one - if the user visited a specific place
visited_location_tags = ["negative_response_visited_LOCATION", "positive_response_visited_LOCATION"]
visited_location_questions = ['Have you ever visited Bucharest?',
                              'Have you ever gone to Iasi?',
                              'Did you ever visit Constanta?',
                              'Did you ever travel to Paris?',
                              'Did you ever travel to Kremlin?',
                              'Did you visit any major city from Romania recently?']

# entity_tags = ["negative_response_ENTITY", "positive_response_ENTITY"]
# entity_questions = ['Have you ever heard of Cher?',
#                     'Have you ever heard of Madonna?',
#                     'Have you ever heard of Maria Tanase?',
#                     'Have you ever heard of Gheorghe Zamfir?',
#                     'Did you ever see Gheorghe Hagi live on television?',
#                     'Did you ever see Nadia Comaneci live on television?',
#                     'Did you ever see Simona Halep live on television?']
initial_question = random.choice(visited_location_questions)
# entity_question = random.choice(entity_questions)

# LOCATION ANSWERS

def dialogue(user_message, number_of_message, response1=None):
    message = user_message
    inattentive = {'typo': 0, 'not_related': 0}
    print("Firstly, you are going to answers some questions about tourist destinations! Relax, there are no right or wrong answers.")
    print(initial_question)

    if number_of_message ==1:
        correct_message, edit_distance = correct_sentence(message) #corrected typos
        message = correct_message
        print(message)
        if edit_distance > 0:
            inattentive['typo'] += edit_distance
        ints = predict_class(message, visited_location_tags)
        if len(ints) > 0:
            res = get_response(ints, intents)
        else:
            inattentive['not_related'] +=1
            res = "Hmm... alright! Can you tell me your best year of your life, according to you?" ## Such an answer is not recognized

        return res


    if number_of_message == 2:
        correct_message, edit_distance = correct_sentence(message) #corrected typos
        message = correct_message
        print(message)
        if edit_distance > 0:
            inattentive['typo'] += edit_distance
        ### combined questions
        if 'LOCATION NAME' in response1 and 'PERSON NAME' in response1:
            location_names = extract_locations(message)
            person_names = extract_people_names(message)
            if len(location_names) ==0 or len(person_names) == 0:
                inattentive['not_related'] += 1
            print(location_names, person_names)

        ### date questions
        elif 'WHEN' in response1 or 'WHAT SEASON' in response1 or 'favourite year' in response1:
            print('2nd extracted')
            dates = extract_dates(message)
            if len(dates) == 0:
                inattentive['not_related'] += 1
            print(dates)

        ### location questions
        elif 'LOCATION NAME' in response1 or 'WHERE' in response1:
            print('3rd extracted')
            location_names = extract_locations(message)
            if len(location_names) == 0:
                inattentive['not_related'] += 1
            print(location_names)

        ### people name questions
        elif 'PERSON' in response1 and 'NAME' in response1:
            print('4th extracted')
            person_names = extract_people_names(message)
            if len(person_names) == 0:
                inattentive['not_related'] += 1
            print(person_names)

        return inattentive




# ENTITY ANSWERS
# print("Now let's move on to questions about local or international stars! There are no right or wrong answers.")
# print(entity_question)
# message_entity = input("")
# correct_message = correct_sentence(message_entity)  # corrected typos
# message_entity = correct_message.term  # only extract the sentence without the edit distance
# print("Message entitiy:", message_entity)
# ints = predict_class(message_entity, entity_tags)
# if len(ints) >0:
#     res = get_response(ints, intents)
# else:
#     res = "Hmm... alright!"
# print(res)
