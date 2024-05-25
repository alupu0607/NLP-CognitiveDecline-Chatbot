import torch
import spacy
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/t5-large-QuestionGeneration')
model = T5ForConditionalGeneration.from_pretrained('Sehong/t5-large-QuestionGeneration')

def generate_questions(text):

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]

    questions = []
    for entity in entities:
        text_with_answer = f"answer:{entity} {text}"
        raw_input_ids = tokenizer.encode(text_with_answer)
        input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

        question_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=512, eos_token_id=1)
        decoded_question = tokenizer.decode(question_ids.squeeze().tolist(), skip_special_tokens=True)
        questions.append(decoded_question)

    return [question.strip("question:") for question in questions]