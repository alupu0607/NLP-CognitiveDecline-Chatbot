import pickle
from flask import Blueprint, jsonify, request
import os
import json
import random

from extractive_question_answering.final_answer_similarity import compute_cosine_similarity
from intent_prediction.chatbot import dialogue, initial_question

bp = Blueprint("pages", __name__)

required_files = [
    r"D:\University\pythonProject1\extractive_question_answering\history_df.pkl",
    r"D:\University\pythonProject1\extractive_question_answering\music_df.pkl",
    r"D:\University\pythonProject1\extractive_question_answering\sport_df.pkl",
    r"D:\University\pythonProject1\extractive_question_answering\history_question_answer_pairs.json",
    r"D:\University\pythonProject1\extractive_question_answering\music_question_answer_pairs.json",
    r"D:\University\pythonProject1\extractive_question_answering\sport_question_answer_pairs.json",
]

# State variables to manage conversation loop
number_of_message = 1
current_question = ""
correct_answer = ""


@bp.route("/")
def home():
    return "Hello, this is my Flask server!"


@bp.before_request
def check_required_files():
    if any(not os.path.exists(file_path) for file_path in required_files):
        print("Required files are missing. Running setup...")
        os.system("python ./extractive_question_answering/extractive_question_answering.py")


@bp.route("/send-message", methods=["POST", "GET"])
def send_message():
    global number_of_message, current_question, correct_answer

    if request.method == "POST":
        try:
            data = request.json
            user_message = data.get("msg", "")

            if number_of_message % 3 == 1:
                response = dialogue(user_message, number_of_message)
                save_response(response)
                number_of_message += 1
                return jsonify({"message": response}), 200

            elif number_of_message % 3 == 2:
                # response = load_response()
                # follow_up = dialogue(user_message, number_of_message, response1=response)
                number_of_message += 1
                return jsonify(
                    {"message": "Thank you! Now it is time for the final question: " + current_question}), 200

            elif number_of_message % 3 == 0:
                received_answer = user_message
                similarity = compute_cosine_similarity(received_answer, correct_answer)
                number_of_message = 1
                return jsonify({"message": f"Similarity score: {similarity}"}), 200

        except Exception as e:
            print("Error processing message:", str(e))
            return jsonify({"error": "Failed to process the message"}), 500

    elif request.method == "GET":
        try:
            category = request.args.get('msg', '').lower()
            file_index = {
                'history category': 3,
                'music category': 4,
                'sports category': 5
            }.get(category)

            if file_index is not None:
                data = get_random_object(required_files[file_index])
                if "error" in data:
                    return jsonify(data), 500

                passage_text = "[...] " + data['answer'][0]['passage_text'] + " [...] " + initial_question
                current_question = preprocess_question(data['question'])
                correct_answer = data['answer'][0]['answer']

                print(f"New passage text for {category}: {passage_text}")
                print(f"New question: {current_question}")
                print(f"Correct answer: {correct_answer}")

                return jsonify({"message": passage_text}), 200
            else:
                return jsonify({"error": "Invalid category"}), 400

        except Exception as e:
            print("Error handling GET request:", str(e))
            return jsonify({"error": "Failed to process the request"}), 500


def save_response(response1):
    with open("response1.pickle", "wb") as f:
        pickle.dump(response1, f)


def load_response():
    with open("response1.pickle", "rb") as f:
        response1 = pickle.load(f)
    return response1


def preprocess_question(question):
    return question.lstrip('?')


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
