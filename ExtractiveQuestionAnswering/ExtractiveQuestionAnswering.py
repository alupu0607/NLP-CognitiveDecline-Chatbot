import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline
from pprint import pprint
import time
from pinecone import Pinecone, ServerlessSpec
import os
from Dataframes import history_df, sport_df, music_df, create_pinecone_database, get_passage_texts
from QuestionGeneration import generate_questions
import json
from dotenv import load_dotenv, find_dotenv

print("ExtractiveQuestionAnswering script is running...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

retriever = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)
batch_size = 64
dataframes = [ history_df, sport_df, music_df]

###PINECONE CONNECTION
load_dotenv(find_dotenv())
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
cloud = os.getenv("CLOUD_KEY")
region = os.getenv("REGION_KEY")
spec = ServerlessSpec(cloud=cloud, region=region)

### HISTORY
index_name_history = "extractive-question-answering-history"

if index_name_history not in pc.list_indexes().names():
    pc.create_index(index_name_history, dimension=384, metric="cosine", spec = spec)
    while not pc.describe_index(index_name_history).status['ready']:
        time.sleep(1)
index_history = pc.Index(index_name_history)
if create_pinecone_database == 1:
    for i in tqdm(range(0, len(history_df), batch_size)):
        i_end = min(i + batch_size, len(history_df))
        batch = history_df.iloc[i:i_end]
        emb = retriever.encode(batch['passage_text'].tolist()).tolist()
        meta = batch.to_dict(orient='records')
        ids = [str(idx) for idx in range(i, i_end)]
        to_upsert = list(zip(ids, emb, meta))
        _ = index_history.upsert(vectors=to_upsert)


### SPORT
index_name_sport = "extractive-question-answering-sport"

if index_name_sport not in pc.list_indexes().names():
    pc.create_index(index_name_sport, dimension=384, metric="cosine", spec = spec)
    while not pc.describe_index(index_name_sport).status['ready']:
        time.sleep(1)
index_sport = pc.Index(index_name_sport)

if create_pinecone_database == 1:
    for i in tqdm(range(0, len(sport_df), batch_size)):
        i_end = min(i + batch_size, len(sport_df))
        batch = sport_df.iloc[i:i_end]
        emb = retriever.encode(batch['passage_text'].tolist()).tolist()
        meta = batch.to_dict(orient='records')
        ids = [str(idx) for idx in range(i, i_end)]
        to_upsert = list(zip(ids, emb, meta))
        _ = index_sport.upsert(vectors=to_upsert)

### MUSIC
index_name_music = "extractive-question-answering-music"

if index_name_music not in pc.list_indexes().names():
    pc.create_index(index_name_music, dimension=384, metric="cosine", spec = spec)
    while not pc.describe_index(index_name_music).status['ready']:
        time.sleep(1)
index_music = pc.Index(index_name_music)

if create_pinecone_database == 1:
    for i in tqdm(range(0, len(music_df), batch_size)):
        i_end = min(i + batch_size, len(music_df))
        batch = music_df.iloc[i:i_end]
        emb = retriever.encode(batch['passage_text'].tolist()).tolist()
        meta = batch.to_dict(orient='records')
        ids = [str(idx) for idx in range(i, i_end)]
        to_upsert = list(zip(ids, emb, meta))
        _ = index_music.upsert(vectors=to_upsert)

#### Model
model_name = 'deepset/electra-base-squad2'
reader = pipeline(tokenizer=model_name, model=model_name, task='question-answering', device=device)

# def get_context(question, top_k):
#     '''
#     This determines from where I get the context -> History
#     '''
#     xq = retriever.encode([question]).tolist()
#     xc = index_history.query(vector=xq, top_k=top_k, include_metadata=True)
#     c = [x["metadata"]['passage_text'] for x in xc["matches"]]
#     return c


def extract_answer(question, passage_text):
    results = []
    answer = reader(question=question, context= passage_text)
    answer["passage_text"] = passage_text
    results.append(answer)
    sorted_result = sorted(results, key=lambda x: x['score'], reverse=True)
    return sorted_result


history_json_path = r"D:\University\pythonProject1\ExtractiveQuestionAnswering\history_question_answer_pairs.json"
sport_json_path = r"D:\University\pythonProject1\ExtractiveQuestionAnswering\sport_question_answer_pairs.json"
music_json_path = r"D:\University\pythonProject1\ExtractiveQuestionAnswering\music_question_answer_pairs.json"
history_json_list = []
sport_json_list = []
music_json_list = []


###Save results
dataframes = [music_df, sport_df]
for df in dataframes:
    passage_texts = get_passage_texts(df)
    for passage_text in passage_texts:
        all_questions = generate_questions(passage_text)

        question_answer_pairs = {}

        for question in all_questions:
            #context = get_context(question, top_k=1)
            answer = extract_answer(question, passage_text)
            question_answer_pairs[question] = answer

        sorted_question_answer_pairs = {question: question_answer_pairs[question] for question, _ in sorted(question_answer_pairs.items(), key=lambda x: x[1][0]['score'], reverse=True)}
        if sorted_question_answer_pairs:
            first_question, first_answer = next(iter(sorted_question_answer_pairs.items()))
        else:
            print("The dictionary is empty.")
        # print("Question:", first_question)
        # print("Answer:", first_answer)

        if df is history_df:
            pair_dict = {
                "type": "History",
                "question": first_question,
                "answer": first_answer
            }
            if len(history_json_list) < 100:
                history_json_list.append(pair_dict)
            else:
                break
            print(pair_dict, len(history_json_list))

        elif df is music_df:
            pair_dict = {
                "type": "Music",
                "question": first_question,
                "answer": first_answer
            }
            if len(music_json_list) < 100:
                music_json_list.append(pair_dict)
            else:
                break
            print(pair_dict, len(music_json_list))

        elif df is sport_df:
            pair_dict = {
                "type": "Sport",
                "question": first_question,
                "answer": first_answer
            }
            if len(sport_json_list) < 100:
                sport_json_list.append(pair_dict)
            else:
                break
            print(pair_dict, len(sport_json_list))


try:
    with open(history_json_path, "a") as history_json_file:
        json.dump(history_json_list, history_json_file, indent=4)
        history_json_file.write("\n")
except Exception as e:
    print(f"Error writing to {history_json_path}: {e}")


try:
    with open(music_json_path, "a") as music_json_file:
        json.dump(music_json_list, music_json_file, indent=4)
        music_json_file.write("\n")
except Exception as e:
    print(f"Error writing to {music_json_path}: {e}")


try:
    with open(sport_json_path, "a") as sport_json_file:
        json.dump(sport_json_list, sport_json_file, indent=4)
        sport_json_file.write("\n")
except Exception as e:
    print(f"Error writing to {sport_json_path}: {e}")