# NLP Cognitive Decline chatbot for Seniority - Elderly Care Platform 
This NLP chatbot is based on a research paper (https://link.springer.com/article/10.1007/s12652-022-03849-2) and uses its proposed solution to identify cognitive decline in seniors.
The chatbot is integrated in the Seniority platform (https://github.com/alupu0607/Seniority).

### Technologies
- Python
- Tensorflow
- Hugging Face transformers
- spacy
- nltk

### About
- A temporary Wikipedia paragraph is shown. A follow-up final question related to this paragraph will be formulated via NLP question-answer pairs generation and a similarity score is calculated. I used NER (Named Entity Recognition) to generate questions as well as Hugging Face transformers. I used `deepset/electra-base-squad-2` to generate the answers. Word embeddings were saved in Pinecone so as to facilitate the answer retrieval via the pipeline.   
- Distractor questions are asked so as to test the short term memory of the user. The user's responses' relevancy is tested via NER (Named Entity Recognition). For example, for a WHEN question the chatbot expects a DATE, for a WITH WHO question the chatbot expects a PERSON. Otherwise, the user will be considered unfocused.
- The chatbot offers empathetic responses via the BoW (Bag of Words) technique. Basically, the chatbot identifies the intent behind a user's response and responds accordingly.

### Status
 Finished

### Authors

Lupu Andreea-Daniela

