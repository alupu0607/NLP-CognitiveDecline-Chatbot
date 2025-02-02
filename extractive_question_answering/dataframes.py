from datasets import load_dataset
import pandas as pd
from tqdm.auto import tqdm
import pickle


def create_dataframe(filter_function,
                     dataset_name='vblagoje/wikipedia_snippets_streamed',
                     total_doc_count = 1500):
    wiki_data = load_dataset(dataset_name, split='train', streaming=True).shuffle(seed=960)
    filtered_data = wiki_data.filter(filter_function)
    docs = []
    for d in tqdm(filtered_data, total=total_doc_count):
        doc = {
            "article_title": d["article_title"],
            "section_title": d["section_title"],
            "passage_text": d["passage_text"]
        }
        docs.append(doc)
        if len(docs) >= total_doc_count:
            break
    df = pd.DataFrame(docs)
    return df


# def get_random_passage_text(dataframe):
#     random_row = dataframe.sample(n=1)
#     random_passage_text = random_row['passage_text'].iloc[0]
#     return random_passage_text

def get_passage_texts(dataframe):
    all_passage_texts = []
    for index, row in dataframe.iterrows():
        passage_text = row['passage_text']
        all_passage_texts.append(passage_text)
    return all_passage_texts


try:
    create_pinecone_database=0
    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\history_df.pkl", "rb") as f:
        history_df = pickle.load(f)
    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\sport_df.pkl", "rb") as f:
        sport_df = pickle.load(f)
    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\music_df.pkl", "rb") as f:
        music_df = pickle.load(f)
except FileNotFoundError:
    create_pinecone_database=1
    history_df = create_dataframe(filter_function=lambda d: d['section_title'].startswith('History'))
    sport_df = create_dataframe(filter_function=lambda d:  'Sport' in d['section_title'] )
    music_df = create_dataframe(filter_function=lambda d:  'Music' in d['section_title'])

    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\history_df.pkl", "wb") as f:
        pickle.dump(history_df, f)
    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\sport_df.pkl", "wb") as f:
        pickle.dump(sport_df, f)
    with open(r"D:\University\pythonProject1\ExtractiveQuestionAnswering\music_df.pkl", "wb") as f:
        pickle.dump(music_df, f)