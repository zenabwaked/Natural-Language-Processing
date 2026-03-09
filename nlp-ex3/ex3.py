# By Malak Laham and Zenab Waked
from collections import Counter, defaultdict
import re
import nltk
from nltk.corpus import brown
import numpy as np

class HMMBigram:
    def __init__(self, train_data, test_data, add_one_smoothing=False, pseudo_smoothing=False, frequency_threshold=2):
        self.frequency_threshold = frequency_threshold
        if pseudo_smoothing:
            self.train_data = list(train_data)
            self.test_data = list(test_data)
        else:
            self.train_data = train_data
            self.test_data = test_data

        # Step 1: Compute original counts
        self.word_to_tag_count, self.tag_to_word_count, self.consecutive_tags_count = self.count_word_tags()

        # Step 2: Apply pseudo-word smoothing if enabled
        if pseudo_smoothing:
            self.replace_with_pseudowords(frequency_threshold)

        # Step 3: Compute the most likely tags
        self.word_to_best_tag = self.calculate_mle_tags()

        # Step 4: Compute error rates
        self.known_error_rate, self.unknown_error_rate, self.total_error_rate = self.compute_error_rates()

        # Step 5: Compute emission and transition probabilities
        self.emission_probabilities = self.compute_emission_probabilities(add_one_smoothing)
        self.transition_probabilities = self.compute_transition_probabilities(add_one_smoothing)

    def process_tag(self, tag):
        """Simplify the tag by removing additional suffixes or prefixes."""
        tag = tag.replace("*", "")
        if "-" in tag:
            tag = tag.split("-")[0]
        if "+" in tag:
            tag = tag.split("+")[0]
        return tag

    def compute_emission_probabilities(self, add_one):
        """Compute emission probabilities P(word | tag)."""
        emission_probabilities = {}
        N = len(self.word_to_tag_count.keys())
        for tag, words in self.tag_to_word_count.items():
            total_tag_appearances = sum(words.values())
            for word, count in words.items():
                if word not in emission_probabilities:
                    emission_probabilities[word] = {}
                if add_one:
                    emission_probabilities[word][tag] = (count + 1) / (total_tag_appearances + N)
                else:
                    emission_probabilities[word][tag] = count / total_tag_appearances
        return emission_probabilities

    def compute_transition_probabilities(self, add_one):
        """Compute transition probabilities P(tag_i | tag_{i-1})."""
        transition_probabilities = {}
        N = len(self.tag_to_word_count.keys())
        for tag, consecutive_tags in self.consecutive_tags_count.items():
            total_tag_appearances = sum(consecutive_tags.values())
            for consecutive_tag, count in consecutive_tags.items():
                if tag not in transition_probabilities:
                    transition_probabilities[tag] = {}
                if add_one:
                    transition_probabilities[tag][consecutive_tag] = (count + 1) / (total_tag_appearances + N)
                else:
                    transition_probabilities[tag][consecutive_tag] = count / total_tag_appearances
        return transition_probabilities

    def viterbi_bigram(self, sentence):
        """Viterbi algorithm for a Bigram Hidden Markov Model using dynamic programming."""
        n = len(sentence)
        tags = list(self.tag_to_word_count.keys())  # List of unique tags
        viterbi = [{} for _ in range(n + 1)]  # DP table for probabilities
        backpointer = [{} for _ in range(n + 1)]  # Backpointer table

        viterbi[0]["*"] = 1
        backpointer[0]["*"] = None

        for k in range(1, n + 1):
            word = sentence[k - 1][0]  # Extract the word (preserve case)
            is_known_word = word in self.emission_probabilities

            for v in tags:
                max_prob = 0
                best_prev_tag = v
                emission_prob = self.emission_probabilities.get(word, {}).get(v, 1 if not is_known_word else 0)
                # emission_prob = self.emission_probabilities.get(word, {}).get(v, 0)
                if emission_prob != 0.0:
                    for u in (tags if k > 1 else ["*"]):
                        prob = (
                            viterbi[k - 1].get(u,  0) *
                            self.transition_probabilities.get(u, {}).get(v, 0) *
                            emission_prob
                        )
                        if prob > max_prob:
                            max_prob = prob
                            best_prev_tag = u
                viterbi[k][v] = max_prob
                backpointer[k][v] = best_prev_tag

        max_final_prob = 0
        best_final_tag = "NN"
        for u in tags:
            prob = viterbi[n].get(u, 0) * self.transition_probabilities.get(u, {}).get("STOP", 0)
            if prob > max_final_prob:
                max_final_prob = prob
                best_final_tag = u

        best_path = [best_final_tag]
        for k in range(n, 0, -1):
            if best_path[0] is None:
                raise ValueError(f"Backtracking failed: backpointer[{k}] contains None.")
            best_path.insert(0, backpointer[k][best_path[0]])
        return best_path[1:]

    def count_word_tags(self):
        """Count all (word, tag) pairs."""
        word_to_tag_counts = defaultdict(Counter)
        tag_to_word_counts = defaultdict(Counter)
        consecutive_tags = defaultdict(Counter)
        for sentence in self.train_data:
            prev = '*'
            for word, tag in sentence:
                tag = self.process_tag(tag)
                word_to_tag_counts[word][tag] += 1
                tag_to_word_counts[tag][word] += 1
                consecutive_tags[prev][tag] += 1
                prev = tag
        return word_to_tag_counts, tag_to_word_counts, consecutive_tags

    def calculate_mle_tags(self):
        """Calculate the most likely tag for each word."""
        word_to_best_tag = {}
        for word, tags in self.word_to_tag_count.items():
            best_tag = max(tags, key=tags.get)
            word_to_best_tag[word] = best_tag
        return word_to_best_tag

    def compute_error_rates(self):
        """Compute error rates for the baseline model."""
        known_correct = 0
        known_total = 0
        unknown_correct = 0
        unknown_total = 0

        train_vocab = set(self.word_to_best_tag.keys())
        for sentence in self.test_data:
            for word, true_tag in sentence:
                true_tag = self.process_tag(true_tag)
                if word in train_vocab:
                    known_total += 1
                    if self.word_to_best_tag[word] == true_tag:
                        known_correct += 1
                else:
                    unknown_total += 1
                    if "NN" == true_tag:
                        unknown_correct += 1

        known_error_rate = 1 - (known_correct / known_total) if known_total > 0 else 0
        unknown_error_rate = 1 - (unknown_correct / unknown_total) if unknown_total > 0 else 0
        total_error_rate = 1 - ((known_correct + unknown_correct) / (known_total + unknown_total))
        return known_error_rate, unknown_error_rate, total_error_rate

    def get_pseudo_word(self,word):
       """
       Generate a pseudo-word for a given word based on its characteristics.
       """
       if re.fullmatch(r"^\$", word):  # Matches words starting with a dollar sign
           return "startsWithDollar"
       elif re.fullmatch(r"\d{2}", word):  # Matches two-digit numbers
          return "twoDigitNum"
       elif re.fullmatch(r"\d{4}", word):  # Matches four-digit numbers
          return "fourDigitNum"
       elif re.search(r"\d", word) and re.search(r"[A-Za-z]", word):  # Contains digits and letters
          return "containsDigitAndAlpha"
       elif re.search(r"\d", word) and "-" in word:  # Contains digits and a dash
          return "containsDigitAndDash"
       elif re.search(r"\d", word) and "/" in word:  # Contains digits and a slash
          return "containsDigitAndSlash"
       elif re.search(r"\d", word) and "," in word:  # Contains digits and a comma
          return "containsDigitAndComma"
       elif re.search(r"\d", word) and "." in word:  # Contains digits and a period
          return "containsDigitAndPeriod"
       elif word.isdigit():  # Matches other numbers
          return "othernum"
       elif word.isupper():  # All uppercase
          return "allCaps"
       elif word.istitle():  # Capitalized (e.g., Proper nouns)
          return "initCap"
       elif re.fullmatch(r"[A-Za-z]\.", word):  # Matches initials like "M."
          return "capPeriod"
       elif word.islower():  # Lowercase
          return "lowercase"
       elif word == ",":  # Specific handling for punctuation
          return "other"
       else:  # All other cases
          return "other"

    def replace_with_pseudowords(self, frequency_threshold=5):
        """
        Replace low-frequency words in the training set and unknown words in the test set with pseudo-words.
        :param frequency_threshold: Words with a total frequency below this value in the training set will be replaced.
        """
        # Step 1: Replace low-frequency words in the training set
        for i, sentence in enumerate(self.train_data):
            for i, (word, tag) in enumerate(sentence):
                # Calculate the frequency of the word by summing its tag counts
                word_frequency = sum(self.word_to_tag_count[word].values()) if word in self.word_to_tag_count else 0
                if word_frequency < frequency_threshold:  # Low-frequency word
                    pseudo_word = self.get_pseudo_word(word)
                    sentence[i] = (pseudo_word, tag)
            self.train_data[i] = sentence
        # Step 3: Recompute counts for word-tag pairs and tag-tag transitions
        self.word_to_tag_count, self.tag_to_word_count, self.consecutive_tags_count = self.count_word_tags()
        # Step 2: Replace unknown words in the test set with pseudo-words
        train_vocab = set(self.word_to_tag_count.keys())
        for i, sentence in enumerate(self.test_data):
            for i, (word, tag) in enumerate(sentence):
                if word not in train_vocab:  # Unknown word
                    # print(word)
                    pseudo_word = self.get_pseudo_word(word)
                    sentence[i] = (pseudo_word, tag)
            self.test_data[i] = sentence


def print_highest_errors(tags, confusion_matrix):
    # Investigate the most frequent errors (excluding diagonal elements)
    errors = []
    for i in range(len(tags)):
        for j in range(len(tags)):
            if i != j:  # Exclude correct predictions (diagonal)
                errors.append((tags[i], tags[j], confusion_matrix[i, j]))
    # Sort errors by frequency
    errors = sorted(errors, key=lambda x: x[2], reverse=True)
    # Print the top 10 most frequent errors
    print("\nMost Frequent Errors:")
    for true_tag, predicted_tag, count in errors[:10]:  # Top 10 errors
        print(f"True Tag: {true_tag}, Predicted Tag: {predicted_tag}, Count: {int(count)}")


def populate_confusion_matrix(sentence, predicted_tags, confusion_matrix, model, tag_to_index):
    for (_, true_tag), predicted_tag in zip(sentence, predicted_tags):
        # Normalize the true tag
        true_tag = model.process_tag(true_tag)
        predicted_tag = model.process_tag(predicted_tag)  # Optional, if needed

        # Ensure normalized tag exists in the index
        if true_tag not in tag_to_index:
            print(f"Skipping unknown true tag: {true_tag}")
            continue

        true_index = tag_to_index[true_tag]
        predicted_index = tag_to_index[predicted_tag]

        # Update confusion matrix
        confusion_matrix[true_index, predicted_index] += 1


def run_b(train_data, test_data):
    model = HMMBigram(train_data, test_data, False, False)
    print(f"\nPart b: evaluate error rates with MLE estimation")
    print(f"Known Words Error Rate: {model.known_error_rate:.4f}")
    print(f"Unknown Words Error Rate: {model.unknown_error_rate:.4f}")
    print(f"Total Error Rate: {model.total_error_rate:.4f}")


def run_part(part, train_data, test_data):
    map_desctiption = {'ciii': 0, 'd': 1, 'eii': 2, 'eiii': 3}
    parts_description = ["Evalute viterbi without Laplace smoothing or pseudo words",
                         "Evaluate viterbi with Laplace smoothing and no pseudo words",
                         "Evaluate viterbi without Laplace smoothing but with replacing pseudowords",
                         "Evaluate viterbi with Laplace smoothing and replacing with pseudowords"]
    description = parts_description[map_desctiption[part]]
    if part == 'ciii':
        model = HMMBigram(train_data, test_data, False, False)
    elif part == 'd':
        model = HMMBigram(train_data, test_data, True, False)
    elif part == 'eii':
        model = HMMBigram(train_data, test_data, False, True)
    elif part == 'eiii':
        model = HMMBigram(train_data, test_data, True, True)
        # List of all tags
        tags = list(model.tag_to_word_count.keys())
        tag_to_index = {tag: i for i, tag in enumerate(tags)}  # Map tags to indices

        # Initialize a numpy confusion matrix
        confusion_matrix = np.zeros((len(tags), len(tags)), dtype=int)

    train_vocab = set(model.word_to_best_tag.keys())
    correct_known, correct_unknown = 0, 0
    num_known_words, num_unknown_words = 0, 0

    for sentence in model.test_data:
        predicted_tags = model.viterbi_bigram(sentence)
        match_list = [1 if predicted == actual[1] else 0 for predicted, actual in zip(predicted_tags, sentence)]
        known_words_list = [1 if word in train_vocab else 0 for word, _ in sentence]
        if part == 'eii' or part == 'eiii':
            # Extract words with frequency <= model.threshold
            low_freq_words = [
                word for word, _ in sentence
                if sum(model.word_to_tag_count.get(word, {}).values()) <= model.frequency_threshold
            ]

        if part == 'eiii':
            populate_confusion_matrix(sentence, predicted_tags, confusion_matrix, model, tag_to_index)

        # Accumulate counts using efficient element-wise operations
        num_known_words += sum(known_words_list)
        num_unknown_words += len(known_words_list) - sum(known_words_list)
        correct_known += sum(m * k for m, k in zip(match_list, known_words_list))
        correct_unknown += sum(m * (1 - k) for m, k in zip(match_list, known_words_list))
    if part == 'eiii':
        print_highest_errors(tags, confusion_matrix)

    # Final calculations
    accuracy_known = correct_known / num_known_words if num_known_words > 0 else 1
    accuracy_unknown = correct_unknown / num_unknown_words if num_unknown_words > 0 else 1
    total_matches = correct_known + correct_unknown
    total_words = num_known_words + num_unknown_words
    accuracy_total = total_matches / total_words if total_words > 0 else 1

    # Error rates
    error_rate_known = 1 - accuracy_known
    error_rate_unknown = 1 - accuracy_unknown
    error_rate_total = 1 - accuracy_total

    # Print results
    print(f"\n Part {part}: {description}")
    print(f"Known Words Error Rate: {error_rate_known:.4f}")
    print(f"Unknown Words Error Rate: {error_rate_unknown:.4f}")
    print(f"Total Error Rate: {error_rate_total:.4f}")


if __name__ == "__main__":
    nltk.download("brown")
    nltk.download("universal_tagset")

    news_tagged_sents = brown.tagged_sents(categories='news')
    split_point = int(0.9 * len(news_tagged_sents))
    train_data = news_tagged_sents[:split_point]
    test_data = news_tagged_sents[split_point:]
    run_b(train_data, test_data)
    run_part('ciii', train_data, test_data)
    run_part('d', train_data, test_data)
    run_part('eii', train_data, test_data)
    run_part('eiii', train_data, test_data)







