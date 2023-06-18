import nltk
from nltk.corpus import brown
import numpy as np

# Collect a corpus of English text
corpus = brown.words()

# Preprocess the text
corpus = [word.lower() for word in corpus if word.isalpha()]

# Tag the text manually
tagged_corpus = brown.tagged_words()

# Define a function to extract features from the text
def pos_features(word):
    # Get the frequency distribution of the word in the corpus
    fd = nltk.FreqDist(corpus)
    # Get the count and frequency of the word in the corpus
    count = fd[word]
    freq = fd.freq(word)
    # Get the seven moment variants of the word's count and frequency
    moments_count = np.array([count, count**2, count**3, count**4, count**5, count**6, count**7])
    moments_freq = np.array([freq, freq**2, freq**3, freq**4, freq**5, freq**6, freq**7])
    # Return the moment variants as features
    return {"count_moment{}".format(i+1): moments_count[i] for i in range(7)} | {"freq_moment{}".format(i+1): moments_freq[i] for i in range(7)}

# Extract features from the tagged corpus
featuresets = [(pos_features(word), pos) for (word, pos) in tagged_corpus]
print(featuresets)
# Split the featuresets into training and testing sets
# train_set, test_set = featuresets[100:], featuresets[:100]

# # Train the POS tagging system using a NaiveBayes classifier
# classifier = nltk.NaiveBayesClassifier.train(train_set)

# # Evaluate the performance of the POS tagging system
# accuracy = nltk.classify.accuracy(classifier, test_set)
# print("Accuracy:", accuracy)

# # Tag some new text with POS labels
# new_text = "The quick brown fox jumps over the lazy dog"
# new_text = new_text.lower().split()
# pos_tags = classifier.classify_many([pos_features(word) for word in new_text])
# print(pos_tags)