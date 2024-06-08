from sentence_transformers import SentenceTransformer, util
def compute_cosine_similarity(sentence1, sentence2, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedding_1 = model.encode(sentence1, convert_to_tensor=True)
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)
    cosine_similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    return cosine_similarity
