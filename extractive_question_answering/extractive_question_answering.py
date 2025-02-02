import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import pipeline
import time
from pinecone import Pinecone, ServerlessSpec
import os
from dataframes import history_df, sport_df, music_df, create_pinecone_database, get_passage_texts
from question_generation import generate_questions
import json
from dotenv import load_dotenv, find_dotenv

print("extractive_question_answering script is running...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sentence embedding model
retriever = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', device=device)
batch_size = 64
dataframes = [history_df, sport_df, music_df]

### Pinecone Connection ###
load_dotenv(find_dotenv())
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
cloud = os.getenv("CLOUD_KEY")
region = os.getenv("REGION_KEY")
spec = ServerlessSpec(cloud=cloud, region=region)


def create_and_index(df, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=384, metric="cosine", spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    index = pc.Index(index_name)

    if create_pinecone_database == 1:
        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i + batch_size, len(df))
            batch = df.iloc[i:i_end]
            emb = retriever.encode(batch['passage_text'].tolist(), batch_size=batch_size).tolist()
            meta = batch.to_dict(orient='records')
            ids = [str(idx) for idx in range(i, i_end)]
            to_upsert = list(zip(ids, emb, meta))
            index.upsert(vectors=to_upsert)
    return index


index_history = create_and_index(history_df, "extractive-question-answering-history")
index_sport = create_and_index(sport_df, "extractive-question-answering-sport")
index_music = create_and_index(music_df, "extractive-question-answering-music")


reader = pipeline(tokenizer='deepset/electra-base-squad2', model='deepset/electra-base-squad2',
                  task='question-answering', device=0 if device == 'cuda' else -1)


def get_context(question, top_k, index):
    question_vector = retriever.encode([question]).tolist()
    results = index.query(vector=question_vector, top_k=top_k, include_metadata=True)
    return [match['metadata']['passage_text'] for match in results['matches']]


def extract_answer(question, index):
    passages = get_context(question, top_k=1, index=index)

    results = []
    for passage_text in passages:
        answer = reader(question=question, context=passage_text)
        answer["passage_text"] = passage_text
        results.append(answer)

    return sorted(results, key=lambda x: x['score'], reverse=True)


def process_and_save(df, index, json_list, json_path, topic_type):
    for passage_text in get_passage_texts(df):
        all_questions = generate_questions(passage_text)
        question_answer_pairs = {}

        for question in all_questions:
            answers = extract_answer(question, index)
            question_answer_pairs[question] = answers

        sorted_question_answer_pairs = {q: question_answer_pairs[q] for q, _ in
                                        sorted(question_answer_pairs.items(), key=lambda x: x[1][0]['score'],
                                               reverse=True)}

        if sorted_question_answer_pairs:
            first_question, first_answer = next(iter(sorted_question_answer_pairs.items()))
            pair_dict = {
                "type": topic_type,
                "question": first_question,
                "answer": first_answer
            }
            json_list.append(pair_dict)
            print(pair_dict)

    try:
        with open(json_path, "w") as json_file:
            json.dump(json_list, json_file, indent=4)
    except Exception as e:
        print(f"Error writing to {json_path}: {e}")


history_json_list = []
sport_json_list = []
music_json_list = []

process_and_save(history_df, index_history, history_json_list, "history_question_answer_pairs.json", "History")
process_and_save(sport_df, index_sport, sport_json_list, "sport_question_answer_pairs.json", "Sport")
process_and_save(music_df, index_music, music_json_list, "music_question_answer_pairs.json", "Music")

print("Processing complete.")
