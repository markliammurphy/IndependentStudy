import heapq
import glob
import re
from random import randint
import numpy as np
import torch
import torch.nn as nn


# read in all reviews to a list of list of words
def read_reviews(data_path, test=False):
    reviews = []
    targets = []
    dir_name = data_path + ('/test' if test else '/train')
    filenames = glob.glob(dir_name + '/pos/*.txt')
    n_pos = len(filenames)
    filenames += glob.glob(dir_name + '/neg/*.txt')
    for i, filename in enumerate(filenames):
        with open(filename) as f:
            words = re.findall(r"[\w]+|[^\s\w(<br />)]", f.read())
            reviews.append([word.lower() for word in words])
            # label positive reviews 1, negative reviews 0
            if i < n_pos:
                targets.append(1)
            else:
                targets.append(0)
    return reviews, targets


# Creates dict of all words to their position
def make_dictionary(reviews):
    dictionary = {}
    for review in reviews:
        for word in review:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


# Gets index of one-hot encoding
def word2index(word, dictionary):
    if word not in dictionary:
        return None
    else:
        return dictionary[word]
    

# Turn review into <review_length x 1 x n_words>
def review2tensor(review, dictionary):
    tensor = torch.zeros(len(review), 1, len(dictionary))
    for wi, word in enumerate(review):
        word_index = word2index(word, dictionary)
        if word_index is not None:
            tensor[wi][0][word_index] = 1
    return tensor


# Turn reviews into list of <review_length x n_words>
def reviews2tensors(reviews, dictionary):
    result = []
    for review in reviews:
        tensor = torch.zeros(len(review), len(dictionary))
        for wi, word in enumerate(review):
            word_index = word2index(word, dictionary)
            if word_index is not None:
                tensor[wi][word_index] = 1
        result.append(tensor)
    return result


# Create the embeddings
def embed_reviews(n_features, dictionary):
    embeds = nn.Embedding(len(dictionary), n_features)
    return embeds


# Turn a list of reviews into list of <review_length x embedding_dim>
def reviews2embeddings(reviews, embeds, dictionary):
    reviews_lookups = [[torch.tensor([dictionary[w]], dtype=torch.long) for w in r] 
               for r in reviews]
    review_tensors = [torch.cat([embeds(lookup) for lookup in lookups], dim=0)
                      for lookups in reviews_lookups]
    return review_tensors
    
# Gets a dictionary of words to the number of times they appear
def get_word_counts(reviews):
    counts = {}
    for review in reviews:
        for word in review:
            if word not in counts:
                counts[word] = 0
            counts[word] += 1
    return counts


# Restricts reviews to only have the top <vocabulary_size> words in them
def preprocess_reviews(reviews, vocabulary_size):
    # construct restricted dictionary
    word_counts = get_word_counts(reviews)
    nth_largest_count = heapq.nlargest(vocabulary_size, word_counts.values())[-1]
    frequent_words = [k for k, v in word_counts.items() if v >= nth_largest_count]
    dictionary = dict(zip(frequent_words, range(vocabulary_size)))
    # filer words from reviews that are not in the dictionary
    processed_reviews = [[word for word in review if word in dictionary]
                         for review in reviews]
    return processed_reviews, dictionary


# Process a set of reviews so that they only have the words in the dictionary
def process_reviews(reviews, dictionary):
    return [[word for word in review if word in dictionary] for review in reviews]


# Select a random training example
def randomTrainingExample(reviews, targets, dictionary):
    i = randint(0, len(targets) - 1)
    target, review = targets[i], reviews[i]
    target_tensor = torch.tensor(target, dtype=torch.float)
    target_tensor = torch.unsqueeze(torch.unsqueeze(target_tensor, 0), 0)
    review_tensor = review2tensor(review, dictionary)
    return target_tensor, review_tensor


# Convert reviews into packed padded tensors
def prepare_batch(reviews, targets, dictionary):
    # Sort reviews and targets by sequence length
    lengths = np.array([len(r) for r in reviews])
    idx = np.argsort(lengths)[::-1]
    sorted_lengths = lengths[idx]
    sorted_reviews = reviews[idx]
    sorted_targets = targets[idx]
    # Pack and zero-pad reviews, convert targets to tensors
    review_tensors = reviews2tensors(sorted_reviews, dictionary)
    padded_reviews = nn.utils.rnn.pad_sequence(review_tensors, batch_first=False)
    batch = nn.utils.rnn.pack_padded_sequence(padded_reviews, sorted_lengths)
    target_tensor = torch.tensor(sorted_targets, dtype=torch.float)
    return target_tensor, batch, sorted_lengths


# Convert reviews into packed padded tensors of word embeddings
def prepare_embedded_batch(reviews, targets, dictionary, embeds):
    # Sort reviews and targets by sequence length
    lengths = np.array([len(r) for r in reviews])
    idx = np.argsort(lengths)[::-1]
    sorted_lengths = lengths[idx]
    sorted_reviews = reviews[idx]
    sorted_targets = targets[idx]
    # Pack and zero-pad reviews, convert targets to tensors
    review_tensors = reviews2embeddings(sorted_reviews, embeds, dictionary)
    padded_reviews = nn.utils.rnn.pad_sequence(review_tensors, batch_first=False)
    batch = nn.utils.rnn.pack_padded_sequence(padded_reviews, sorted_lengths)
    target_tensor = torch.tensor(sorted_targets, dtype=torch.float)
    return target_tensor, batch, sorted_lengths
