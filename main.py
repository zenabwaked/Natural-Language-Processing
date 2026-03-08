import math
import spacy
from datasets import load_dataset


def process_string(str_to_process):
    doc = nlp(str_to_process)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    return ["START"] + tokens


def preprocess_docs(text):
    """
    Preprocess the text to generate lemmatized tokens and handle per-document logic.

    Args:
        text (list of str): List of lines from the dataset.

    Returns:
        list of list: A list of lemmatized token lists for each document.
    """
    docs = []
    for line in text:
        tokens = process_string(line)
        if tokens == ["START"]:
            continue
        docs.append(tokens)  # Add START token
    return docs

#
#Task-1
def train_unigram(docs):
    token_counts = {}
    for doc in docs:
        for token in doc[1:]:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

    total_tokens_count = 0
    for count in token_counts.values():
        total_tokens_count += count

    unigram_probs = {}
    for token, count in token_counts.items():
        unigram_probs[token] = math.log(count / total_tokens_count)

    return unigram_probs


def train_bigram(doc):
    pair_counts = {}
    word_counts = {}
    for doc in docs:
        for i in range(len(doc) - 1):
            pair = (doc[i], doc[i + 1])
            if pair in pair_counts:
                pair_counts[pair] += 1
            else:
                pair_counts[pair] = 1

            if doc[i] in word_counts:
                word_counts[doc[i]] += 1
            else:
                word_counts[doc[i]] = 1

        if doc[-1] in word_counts:
            word_counts[doc[-1]] += 1
        else:
            word_counts[doc[-1]] = 1

    bigram_probs = {}
    for bigram, count in pair_counts.items():
        first_word = bigram[0]
        bigram_probs[bigram] = math.log(count / word_counts[first_word])

    return bigram_probs


#Task-2
def predict_bigram(processed_sentence, bigram_probs):
    """
    Predict the next word based on bigram probabilities.

    Args:
        processed_sentence (str): The input sentence fragment.
        bigram_probs (dict): Dictionary of bigram probabilities.

    Returns:
        str: The predicted next word or a message indicating no prediction.
    """
    last_word = processed_sentence[-1]
    candidates = {pair[1]: prob for pair, prob in bigram_probs.items() if pair[0] == last_word}
    if not candidates:
        return "No prediction available (no matching next word)."
    predicted_word = max(candidates, key=candidates.get)
    return predicted_word


#Task-3
def compute_sentence_prob_bigram(processed_sentence , bigram_probs):
    """
    Compute the probability of a sentence using the bigram model.

    Args:
        processed_sentence (str): The input sentence.
        bigram_probs (dict): Bigram probabilities.

    Returns:
        float: Probability of the sentence.
    """
    result = 0
    for i in range(len(processed_sentence)-1):
        pair = (processed_sentence[i], processed_sentence[i + 1])
        if pair in bigram_probs:
            result += bigram_probs[pair]
        else:
            return -math.inf
    return result


def compute_bigram_preplexity(processed_sentences, bigram_probs):
    M = 0
    probability_sum = 0
    for sentence in processed_sentences:
        M += len(sentence[1:])
        probability_sum += compute_sentence_prob_bigram(sentence, bigram_probs) # this returns logP(si)
    return math.exp(- probability_sum / M)


#Task-4
def compute_interpolated_pair_probability(token1, token2, unigram_probs, unigram_weight, bigram_probs, bigram_weight):
    P_unigram = math.exp(unigram_probs.get(token2, float('-inf')))
    P_bigram = math.exp(bigram_probs.get((token1, token2), float('-inf')))
    interpolated_prob = bigram_weight * P_bigram + unigram_weight * P_unigram
    return math.log(interpolated_prob)


def compute_sentence_probability_with_interpolation(processed_sentence, unigram_probs,unigram_weight, bigram_probs, bigram_weight):
    probability_sum = 0
    for i in range(len(processed_sentence) - 1):
        prob = compute_interpolated_pair_probability(processed_sentence[i], processed_sentence[i + 1], unigram_probs, unigram_weight, bigram_probs, bigram_weight)
        probability_sum += prob
    return probability_sum


def compute_interpolated_perplexity(processed_sentences, unigram_probs,unigram_weight, bigram_probs, bigram_weight):
    M = 0
    probability_sum = 0
    for sentence in processed_sentences:
        M += len(sentence[1:])
        probability_sum += compute_sentence_probability_with_interpolation(sentence, unigram_probs,unigram_weight, bigram_probs, bigram_weight)
    return math.exp(-probability_sum / M)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    text = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")
    docs = preprocess_docs(text['text'])

    #Task-1
    unigram_probs = train_unigram(docs)
    bigram_probs = train_bigram(docs)

    #Task-2
    print("Task 2:")
    print("I have a house in " + predict_bigram(process_string("I have a house in "), bigram_probs))

    #Task-3-a
    sentence1 = "Brad Pitt was born in Oklahoma"
    sentence2 = "The actor was born in USA"
    processed_sentence1 = process_string(sentence1)
    processed_sentence2 = process_string(sentence2)

    prob1 = compute_sentence_prob_bigram(processed_sentence1, bigram_probs)
    prob2 = compute_sentence_prob_bigram(processed_sentence2, bigram_probs)

    print("Task 3-a:")
    print(f"Bigram probability for '{sentence1}': {prob1:.3f}")
    print(f"Bigram probability for '{sentence2}': {prob2:.3f}")

    #Task-3-b
    print("Task 3-b:")
    print("The perplexity of both sentences: " + str(compute_bigram_preplexity([processed_sentence1, processed_sentence2], bigram_probs)))

    #Task-4
    print("Task 4:")
    print("Linear Interpolation with weights 1/3 and 2/3")
    unigram_weight = 1/3
    bigram_weight = 2/3

    interpolation_prob1 = compute_sentence_probability_with_interpolation(processed_sentence1, unigram_probs, unigram_weight, bigram_probs, bigram_weight)
    interpolation_prob2 = compute_sentence_probability_with_interpolation(processed_sentence2, unigram_probs, unigram_weight, bigram_probs, bigram_weight)
    print(f"The probability for '{sentence1}': {interpolation_prob1:.3f}")
    print(f"The probability for '{sentence2}': {interpolation_prob2:.3f}")

    interp_perplexity = compute_interpolated_perplexity([processed_sentence1, processed_sentence2], unigram_probs, unigram_weight, bigram_probs, bigram_weight)
    print(f"The interpolated perplexity of both sentences: {interp_perplexity:.3f}")
