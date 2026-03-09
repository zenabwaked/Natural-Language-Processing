import random
import wikipedia
import google.generativeai as genai
import spacy
nlp = spacy.load("en_core_web_sm")

PUNCT_INDEX = 0
VERB_INDEX = 1
SEQUENCE_INDEX = 2


def find_proper_nouns(doc):
    proper_nouns = []
    consec_pairs_propns = []
    prev_is_propn = False

    for token in doc:
        if token.pos_ == 'PROPN':
            if prev_is_propn:
                proper_nouns[-1].append(token.text)
            else:
                proper_nouns.append([token.text])
                consec_pairs_propns.append([False, False, []])
            prev_is_propn = True
        else:
            prev_is_propn = False
            if len(proper_nouns):
                if token.pos_ == 'PUNCT':
                    consec_pairs_propns[-1][PUNCT_INDEX] = True
                elif token.pos_ == 'VERB':
                    consec_pairs_propns[-1][VERB_INDEX] = True
                    consec_pairs_propns[-1][SEQUENCE_INDEX].append(token.text)
                elif token.pos_ == 'ADP':
                    consec_pairs_propns[-1][SEQUENCE_INDEX].append(token.text)

    return proper_nouns, consec_pairs_propns


class SroTriplet:
    def __init__(self):
        self.subject = ''
        self.relation = ''
        self.object = ''

    def __str__(self):
        return f"({self.subject}, {self.relation}, {self.object})"


def find_subject_relation_object_pos(doc):
    proper_nouns, consec_pairs_propns = find_proper_nouns(doc)
    triplets = []

    for i, pair in enumerate(consec_pairs_propns):
        if not pair[PUNCT_INDEX] and pair[VERB_INDEX]:  # valid triplet condition
            triplet = SroTriplet()
            triplet.subject = " ".join(proper_nouns[i])
            triplet.relation = " ".join(pair[SEQUENCE_INDEX])
            triplet.object = " ".join(proper_nouns[i + 1]) if i + 1 < len(proper_nouns) else ""
            triplets.append(str(triplet))

    return triplets


def propn_head_extractor(doc):
    proper_noun_sets = []
    proper_noun_heads = []

    for token in doc:
        if token.pos_ == "PROPN" and token.dep_ != "compound":
            proper_noun_set = {token.text}
            proper_noun_heads.append(token)

            for child in token.children:
                if child.dep_ == "compound":
                    proper_noun_set.add(child.text)

            proper_noun_sets.append(proper_noun_set)

    return proper_noun_sets, proper_noun_heads


def get_full_name(token):
    full_name = [token.text]
    for child in token.children:
        if child.dep_ == "compound":
            full_name.insert(0, child.text)
    return " ".join(full_name)


def find_subject_relation_object_trees(doc):
    proper_noun_sets, proper_noun_heads = propn_head_extractor(doc)
    triplets = []

    for h1 in proper_noun_heads:
        for h2 in proper_noun_heads:
            if h1 != h2:
                h = h1.head
                h0 = h2.head

                # Direct subject-verb-object triplet
                if h == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                    triplet = SroTriplet()
                    triplet.subject = get_full_name(h1)
                    triplet.relation = h.text
                    triplet.object = get_full_name(h2)
                    triplets.append(str(triplet))

                # Prepositional phrase triplet
                if h == h0.head and h1.dep_ == "nsubj" and h0.dep_ == "prep" and h2.dep_ == "pobj":
                    triplet = SroTriplet()
                    triplet.subject = get_full_name(h1)
                    triplet.relation = h.text + " " + h0.text
                    triplet.object = get_full_name(h2)
                    triplets.append(str(triplet))

                # Passive voice triplet
                if h1.dep_ == "nsubjpass" and h2.dep_ == "agent":
                    triplet = SroTriplet()
                    triplet.subject = get_full_name(h2)
                    triplet.relation = h.text
                    triplet.object = get_full_name(h1)
                    triplets.append(str(triplet))

                # Indirect object triplet
                if h1.dep_ == "nsubj" and h2.dep_ == "iobj":
                    triplet = SroTriplet()
                    triplet.subject = get_full_name(h1)
                    triplet.relation = h.text
                    triplet.object = get_full_name(h2)
                    triplets.append(str(triplet))

    return triplets


# Load Gemini API
genai.configure(api_key="REMOVED")


def query_gemini(text):
    model = genai.GenerativeModel("gemini-pro")

    prompt = (
        "Extract subject-relation-object (SRO) triplets where:"
        "- The **Subject** and **Object** must be proper nouns (names, locations, or entities)."
        "- The **Relation** must be a single verb (or a verb followed by a preposition)."
        "- The Object must **not** include the verb or prepositions, just the object."
        "Format each triplet as (Subject, Relation, Object)."
        "For example:\n"
        " - ('Alice', 'dropped', 'the mic')\n"
        " - ('Jenna', 'apologized to', 'Camilla')\n"
        " - ('Trump', 'withheld', 'military aid')\n"
        " - ('Trump', 'granted', '237 requests')\n"
        f"Here is the text to analyze:\n{text}"
    )

    response = model.generate_content(prompt)

    if hasattr(response, "text"):
        triplets = response.text.strip().split("\n")
        return triplets
    else:
        return ["Error: Gemini API returned an invalid response"]


def random_sample_triplets(triplets, sample_size=5):
    """Randomly selects a sample of triplets for manual validation."""
    if len(triplets) < sample_size:
        return triplets  # Return all if less than sample size
    return random.sample(triplets, sample_size)


def evaluate_extractor_pos(doc, page_name):
    triplets_pos = find_subject_relation_object_pos(doc)
    triplets_tree = find_subject_relation_object_trees(doc)
    len_text = len(doc.text)
    first_set = doc.text[:int(len_text/3)]
    second_set = doc.text[int(len_text/3):2*int(len_text/3)]
    third_set = doc.text[2*int(len_text/3):]
    llm_output_1 = query_gemini(first_set)
    llm_output_2 = query_gemini(second_set)
    llm_output_3 = query_gemini(third_set)
    llm_output_1.extend(llm_output_2)
    llm_output_1.extend(llm_output_3)

    # Randomly select triplets for validation
    sample_pos = random_sample_triplets(triplets_pos, 5)
    sample_tree = random_sample_triplets(triplets_tree, 5)
    sample_llm = random_sample_triplets(llm_output_1, 5)

    print(f"-------------  Wikipedia page of {page_name}  -------------")
    print(f"For POS tags extractor the number of triplets is: {len(triplets_pos)}")
    print(f"For the dependency tree extractor the number of triplets is: {len(triplets_tree)}")
    print(f"LLM Extractor - Number of triplets: {len(llm_output_1)}")

    print("Random Samples for Manual Validation")

    print("## POS-based Extractor Samples:")
    for triplet in sample_pos:
        print(f"   {triplet}")

    print("## Dependency Tree Extractor Samples:")
    for triplet in sample_tree:
        print(f"   {triplet}")

    print("## LLM Extractor Samples:")
    for triplet in sample_llm:
        print("   ("+ f"{triplet}" +")")

    print("\n-------------------------------------------------------------------\n")

wiki_pages = ['Donald Trump', 'Ruth Bader Ginsburg', 'J.K.Rowling']
for title in wiki_pages:
    page = wikipedia.page(title).content
    analyzed_page_doc = nlp(page)
    evaluate_extractor_pos(analyzed_page_doc, title)