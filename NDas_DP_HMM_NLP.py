#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:31:19 2019

@author: Nishant Das 

Language Models calculate the probability of a sequence of words. 
With this we can model sentences and determine if the a sentence is correct or incorrect given the probability of sequences of words in that given sentence.
If the probability of the sequence of words in a sentence follows a distribution typical of the language (English in this case), we can classify a sentence as being correct or incorrect, or real or not real.

For this project, I have created an algorithm to determine if a sentence given by a user is correct or incorrect. 

Words typically are used in association with other words. 
This is typical when languages have a structure (Noun followed by a verb, etc.). 
Hence, by looking at word(t) and word(t+1) occurrences (adjacent word occurrences), we can see how likely a sentence is real or not. 

Bigram is a sequence of two consecutive words in a sentence. 

In this project, I use the Brown Corpus (explained in the code development section) of words to train a Hidden Markov Model that is a variation of Viterbi Algorithm.
The Viterbi Algorithm is a Dynamic Programming algorithm to the the most likely sequence of hidden states – known as the Viterbi path. 
Given the model, we can then pass English sentences into the model to predict if the sentence is an actual English sentence or not. 

The Brown Corpus 500 samples of English-language text, totaling roughly one million words, compiled from works published in the United States in 1961.




"""
# Importing relevent libraries
from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
import numpy as np
# Importing nltk and brown corpus
import nltk
# Below line is a one time step - disabled after downloading
nltk.download('brown')
from nltk.corpus import brown


def get_sentences():
  # returns 57340 words of the Brown corpus in the form of sentences taken from various sources. 
  # each sentence is represented as a list of words represented as individual string words
  return brown.sents()

def get_sentences_with_wordVecIds():
  sentences = get_sentences()
  indexed_sentences = []
  i = 2
  word2idx = {'START': 0, 'END': 1} #Made up words to signify start and end of a sentence. 
  for sentence in sentences:
    indexed_sentence = []
    for word in sentence:
      word = word.lower()
      if word not in word2idx:
        word2idx[word] = i
        i += 1
      indexed_sentence.append(WordVecIds[word])
    indexed_sentences.append(indexed_sentence)
  print("Vocab size:", i)
  return indexed_sentences, wordVecIds

KEEP_WORDS = set([
    # Any words we may need to include in our Word Vector. 
])
    
def get_sentences_with_wordVecIds_limit_vocab(n_vocab=3000, keep_words=KEEP_WORDS):
  sentences = get_sentences()
  indexed_sentences = []

  i = 2

# Initializing Relevent Dictionaries
# Word vector in Id Number format and Word Format.
# First and Second words of the Word vector are START & END
  wordVecIds = {'START': 0, 'END': 1}
  wordVecWords = ['START', 'END']

# Counter to count hoew many times a word occurs in our Courpus
  word_idx_count = {
    0: float('inf'), #Start & End are infinite counts
    1: float('inf'),
  }

  for sentence in sentences:
    indexed_sentence = []
    for word in sentence:
      word = word.lower()
      if word not in wordVecIds:
        wordVecWords.append(word) # Add the word to the word vector as the 3rd word
        wordVecIds[word] = i # Initalize the count for that word in the word vector as the 3rd word
        i += 1

      # keep track of counts of a word for later sorting
      idx = wordVecIds[word] #index of a word
      word_idx_count[idx] = word_idx_count.get(idx,0) + 1 

      indexed_sentence.append(idx) # Building the numbered sentences
    indexed_sentences.append(indexed_sentence) 

# restrict vocab size
  # set all the words I want to keep to infinity
  # so that they are included when the words are sorted and filtered by count 
  # common words
  for word in keep_words:
    word_idx_count[wordVecIds[word]] = float('inf')


    
  sorted_word_idx_count = sorted(word_idx_count.items(), key=lambda elem : elem[1], reverse=True) #Sorting in descending order
  word2idx_small = {}
  new_idx = 0
  idx_new_idx_map = {} #Upadating the ids of words in descending order to word vector in ID Number format
  for idx, count in sorted_word_idx_count[:n_vocab]: #for words upto the user defined numbber of words
    word = wordVecWords[idx]
    #print(word, count)
    word2idx_small[word] = new_idx
    idx_new_idx_map[idx] = new_idx
    new_idx += 1
  # let 'unknown' be the last word
  word2idx_small['UNKNOWN'] = new_idx 
  unknown = new_idx

  assert('START' in word2idx_small)
  assert('END' in word2idx_small)
  for word in keep_words:
    assert(word in word2idx_small)

  # map old idx to new idx
  sentences_small = []
  for sentence in indexed_sentences:
    if len(sentence) > 1:
      new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
      sentences_small.append(new_sentence)

  return sentences_small, word2idx_small


def get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=1):
  # structure of bigram probability matrix will be:
  # (last word, current word) --> probability
  # we will use add 1 smoothing
  # note: we'll always ignore this from the END token
  bigram_probs_matrix = np.ones((V, V)) * smoothing # Matrices of 1's times smoothing, V is the number of words in the word vector. 
  for sentence in sentences:
    for i in range(len(sentence)):
      
      if i == 0:
        # beginning word
        bigram_probs_matrix[start_idx, sentence[i]] += 1 # Count of Start w(t)
      else:
        # middle word
        bigram_probs_matrix[sentence[i-1], sentence[i]] += 1 # Counting occurances of w(t-1) w(t)
      # if we're at the final word
      # we update the bigram for last -> current
      # AND current -> END token
      if i == len(sentence) - 1: 
        # final word 
        bigram_probs_matrix[sentence[i], end_idx] += 1 # Count of w(t) End

  # normalize the counts along the rows to get probabilities
  bigram_probs_matrix /= bigram_probs_matrix.sum(axis=1, keepdims=True) #Sum(Summation(w(t(i)-1)*w(t)/(all w(t-1)w(t)) 
    #combinations ffor a given w(t) to get the probabilties Matrix
  return bigram_probs_matrix

if __name__ == '__main__':
  # load in the data
  # note: sentences are already converted to sequences of word indexes
  # note: you can limit the vocab size if you run out of memory
  sentences, wordVecIds = get_sentences_with_wordVecIds_limit_vocab(10000)
  # sentences, word2idx = get_sentences_with_word2idx()

  # vocab size
  V = len(wordVecIds)
  #print("Vocab size:", V)
  # we will also treat beginning of sentence and end of sentence as bigrams
  # START -> first word
  # END -> last word 
    #Defining the start and end index numbers
  start_idx = wordVecIds['START']
  end_idx = wordVecIds['END']

# Markov Model
  # a matrix where:
  # row = last word
  # col = current word
  # value at [row, col] = p(current word | last word)
  bigram_probs_matrix = get_bigram_probs(sentences, V, start_idx, end_idx, smoothing=0.1)
  # a function to calculate normalized log prob score for a user defined sentence
  def get_score(sentence):
    score = 0
    for i in range(len(sentence)):
      if i == 0:
        # For the first word, calculate the Start Word occurance matrix and take the log to update the score.
        score += np.log(bigram_probs_matrix[start_idx, sentence[i]])
      else:
        # Do the same for all words in the sentence tille the final word
        # middle word
        score += np.log(bigram_probs_matrix[sentence[i-1], sentence[i]])
    # final word
    score += np.log(bigram_probs_matrix[sentence[-1], end_idx])

    # normalize the final score by divding by the number of words + 1
    return score / (len(sentence) + 1)


  # a function to map word indexes back to real words
  idx2word = dict((v, k) for k, v in iteritems(wordVecIds))
  def get_words(sentence):
        for i in sentence:
            return ' '.join(idx2word[i])
                                            ########################return ' '.join( for i in sentence)

# Generating some fake sentences randomly

  # when we generate fake sentence, we want to ensure not to use
  # start -> word or end -> word combinations
  sample_probs = np.ones(V) # Matrix of ones
  sample_probs[start_idx] = 0 #Set start and end combninations to zero
  sample_probs[end_idx] = 0
  sample_probs /= sample_probs.sum() #


# User Interface

  # test our model on real and fake sentences
  while True:
    
    # input your own sentence
    custom = input("Enter your own sentence:\n\n")
    custom = custom.lower().split()

    # check that all words exist in wordVecIds (otherwise, we can't get score)
    bad_sentence = False
    for word in custom:
      if word not in wordVecIds:
        bad_sentence = True

    if bad_sentence:
      print("Sorry, you entered words that are not in the Corpus Vocabulary")
    else:
      # convert sentence into list of indexes
      custom = [wordVecIds[word] for word in custom]
      print("SCORE:", get_score(custom))
      if get_score(custom)>(-8.2): # Decided on this number after various trials - could possibily right a classification model for this. 
        print("REAL SENTENCE")
      else:
        print("NOT A REAL SENTENCE")


    cont = input("Continue? [Y/n]")
    if cont and cont.lower() in ('N', 'n'):
        print("Thank You!")
        break
    

