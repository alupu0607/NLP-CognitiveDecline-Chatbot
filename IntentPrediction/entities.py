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

