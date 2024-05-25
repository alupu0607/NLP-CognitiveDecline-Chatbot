import pickle

from flask import Blueprint, jsonify, request
import os
import json
import random

from ExtractiveQuestionAnswering.FinalAnswerSimilarity import compute_cosine_similarity
from IntentPrediction.chatbot import dialogue, initial_question

bp = Blueprint("pages", __name__)
required_files = [
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\history_df.pkl",
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\music_df.pkl",
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\sport_df.pkl",
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\history_question_answer_pairs.json",
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\music_question_answer_pairs.json",
        r"D:\University\pythonProject1\ExtractiveQuestionAnswering\sport_question_answer_pairs.json",
    ]

@bp.route("/")
def home():
    return "Hello, this is my Flask server!"

@bp.before_request
def check_history_file():

    if any(not os.path.exists(file_path) for file_path in required_files):
        print("is running...")
        os.system("python ./ExtractiveQuestionAnswering/ExtractiveQuestionAnswering.py")

number_of_message = 1
question = ""
answer = ""
@bp.route("/send-message", methods=["POST", "GET"])
def send_message():
    global question, answer
    if request.method == "POST":
        try:
            data = request.json
            #print("Received message:", data.get("msg"))
            global number_of_message
            if number_of_message == 1:
                response1 = dialogue(data.get("msg"), number_of_message)
                save_response(response1)
                number_of_message += 1
                return jsonify({"message": response1}), 200
            elif number_of_message == 2:
                response1 = load_response()
                inattentive = dialogue(data.get("msg"), number_of_message, response1= response1)
                print(inattentive)
                number_of_message += 1
                return jsonify({"message": "Thank you! Now it is time for the final question..." + question}), 200
            else:
                received_answer = data.get("msg")
                print(answer)
                similarity = compute_cosine_similarity(received_answer, answer)
                print(similarity)
                return jsonify({"message": "Done"}), 200
        except Exception as e:
            print("Error processing message:", str(e))
            return jsonify({"error": "Failed to process the message"}), 500
    elif request.method == "GET":
        if 'msg' in request.args and request.args.get('msg') == 'History category':
            data = get_random_object(required_files[3])
            print("init question", initial_question)
            passage_text = "[...] " + data['answer'][0]['passage_text'] + " [...]" + initial_question
            question = preprocess_question(data['question'])
            answer = data['answer'][0]['answer']
            print(question, answer)
            return jsonify({"message": passage_text}), 200



        elif 'msg' in request.args and request.args.get('msg') == 'Music category':
            data = get_random_object(required_files[4])

            passage_text = "[...] " + data['answer'][0]['passage_text'] + " [...]" + initial_question
            return jsonify({"message": passage_text}), 200
        elif 'msg' in request.args and request.args.get('msg') == 'Sports category':
            data = get_random_object(required_files[5])
            passage_text = "[...] " + data['answer'][0]['passage_text'] + " [...]"
            return jsonify({"message": passage_text}), 200
        else:
            return jsonify({"error": "Invalid category"}), 400

def save_response(response1):
    with open("response1.pickle", "wb") as f:
        pickle.dump(response1, f)

# Function to load response1
def load_response():
    with open("response1.pickle", "rb") as f:
        response1 = pickle.load(f)
    return response1
def preprocess_question(question):
    if question.startswith('?'):
        question = question[1:]
    return question
def get_random_object(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if isinstance(data, list) and data:
                return random.choice(data)
            else:
                return {"error": "File does not contain a valid list of JSON objects"}
    except FileNotFoundError:
        return {"error": "File not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in the file"}