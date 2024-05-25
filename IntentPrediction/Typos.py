
from symspellpy import SymSpell
import Levenshtein as lev
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "frequency_dictionary_en_82_765.txt"
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
