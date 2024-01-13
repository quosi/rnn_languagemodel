import nltk
from nltk import ngrams
from collections import defaultdict
import random


def read_file(filepath):
    with open(filepath) as whole_text:
        corpus = whole_text.read().lower()
    return corpus


def create_tokens(text, n=2):
    # Define the order of the N-gram model
    # N=2 for bigrams, N=3 for trigrams etc.
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Preprocess the words (convert to lowercase, remove punctuations)
    words = [word.lower() for word in words if word.isalnum()]

    # Create N-grams from the tokenized words
    ngrams_list = list(ngrams(words, n))
    # Create a default dict to store N-grams and their frequency
    # eliminating duplicate appearances of word tuples at the same time
    ngram_dict = defaultdict(int)
    for ngram in ngrams_list:
        ngram_dict[ngram] += 1

    return ngram_dict


# Define Function
def predict_next_word_quick_print(prefix, ngram_freq):
    # Filter N-grams that start with the given prefix
    matching_ngrams = [(ngram, freq) for ngram, freq in ngram_freq.items() if ngram[:-1] == prefix]

    if not matching_ngrams:
        return "No prediction available."

    # Sort N-grams by frequency in descending order,
    # so N-gram with the highest frequency is at the top of prediction list,
    # hence most likely to be predicted, because of highest rating
    sorted_ngrams = sorted(matching_ngrams, key=lambda x: x[1], reverse=True)
    # to return highest scored next word prediction use -> sorted_ngrams[0][0][-1]
    p_nr = random.randint(0, len(sorted_ngrams) - 1)
    return sorted_ngrams[p_nr][0][0].title() + " " + sorted_ngrams[p_nr][0][-1].title()


def predict_next_word(prefix, ngram_freq):
    # Filter N-grams that start with the given prefix
    matching_ngrams = [(ngram, freq) for ngram, freq in ngram_freq.items() if ngram[:-1] == prefix]
    if not matching_ngrams:
        return "No prediction available."
    # Sort N-grams by frequency in descending order,
    sorted_ngrams = sorted(matching_ngrams, key=lambda x: x[1], reverse=True)

    p = random.randint(0, len(sorted_ngrams) - 1)
    return (sorted_ngrams[p][0][-1],)


def predict_n_words(prefix, ngram_freq, nr_p=1):
    prefix_lst = [prefix]
    for i in range(nr_p):
        pred_word = predict_next_word(prefix_lst[-1], ngram_freq)
        prefix_lst.append(pred_word)
    return ' '.join(map(str, [x[0] + " " for x in prefix_lst])).capitalize()


if __name__ == '__main__':
    # Interactively test the model with user input
    user_input = input("Enter a prefix for next-word prediction: ").lower().split()
    N = 2
    text_input = read_file('data/Drei-Meister-Balzac-Dickens-Dostojewski_Stefan-Zweig_Projekt-Gutenberg_36389-8.txt')
    ngram_freq = create_tokens(text_input, N)

    if len(user_input) != N - 1:
        print("Please enter a valid prefix.")
    else:
        prefix = tuple(user_input)
        prediction = predict_next_word_quick_print(prefix, ngram_freq)
        print(f"Next word prediction: {prediction}")
        nr = 6
        print(f"{nr} words prediction: {predict_n_words(prefix, ngram_freq, nr_p=nr)}")
