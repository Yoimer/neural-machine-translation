# This tutorial is to Develop a Deep Learning Model to Automatically
# Translate from German to English in Python with Keras, Step-by-Step.
# Machine translation is a challenging task that traditionally involves large statistical models 
# developed using highly sophisticated linguistic knowledge.

# Neural machine translation is the use of deep neural networks for the problem of machine translation.

# In this tutorial, you will discover how to develop a neural machine translation system for translating German phrases to English.

# After completing this tutorial, you will know:

# How to clean and prepare data ready to train a neural machine translation system.
# How to develop an encoder-decoder model for machine translation.
# How to use a trained model for inference on new input phrases and evaluate the model skill.

# Tutorial Overview
# This tutorial is divided into 4 parts; they are:

# German to English Translation Dataset
# Preparing the Text Data
# Train Neural Translation Model
# Evaluate Neural Translation Model
# Python Environment
# This tutorial assumes you have a Python 3 SciPy environment installed.

# You must have Keras (2.0 or higher) installed with either the TensorFlow or Theano backend.

# The tutorial also assumes you have NumPy and Matplotlib installed.

# If you need help with your environment, see this post:

# How to Setup a Python Environment for Machine Learning and Deep Learning with Anaconda

# https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

# A GPU is not require for thus tutorial, nevertheless, you can access GPUs cheaply on Amazon Web Services. Learn how in this tutorial:

# https://machinelearningmastery.com/develop-evaluate-large-deep-learning-models-keras-amazon-web-services/

# How to Setup Amazon AWS EC2 GPUs to Train Keras Deep Learning Models (step-by-step)
# Let’s dive in.

# German to English Translation Dataset
# In this tutorial, we will use a dataset of German to English terms used as the basis for flashcards for language learning.

# The dataset is available from the ManyThings.org website, with examples drawn from the Tatoeba Project. The dataset is comprised of German phrases and their English counterparts and is intended to be used with the Anki flashcard software.

# The page provides a list of many language pairs, and I encourage you to explore other languages:

# Tab-delimited Bilingual Sentence Pairs
# The dataset we will use in this tutorial is available for download here:

# German – English deu-eng.zip
# http://www.manythings.org/anki/deu-eng.zip

# Download the dataset to your current working directory and decompress; for example:

!wget http://www.manythings.org/anki/deu-eng.zip

!unzip deu-eng.zip # remeber to open a notebook in google colab

# You will have a file called deu.txt that contains 152,820 pairs of English to German phases, one pair per line with a tab separating the language.

# For example, the first 5 lines of the file look as follows:

# Hi. Hallo!
# Hi. Grüß Gott!
# Run!    Lauf!
# Wow!    Potzdonner!
# Wow!    Donnerwetter!
# 1
# 2
# 3
# 4
# 5
# Hi. Hallo!
# Hi. Grüß Gott!
# Run!    Lauf!
# Wow!    Potzdonner!
# Wow!    Donnerwetter!


# We will frame the prediction problem as given a sequence of words in German as input, translate or predict the sequence of words in English.

# The model we will develop will be suitable for some beginner German phrases.

# Preparing the Text Data
# The next step is to prepare the text data ready for modeling.

# If you are new to cleaning text data, see this post:

# How to Clean Text for Machine Learning with Python
# https://machinelearningmastery.com/clean-text-machine-learning-python/

# Take a look at the raw data and note what you see that we might need to handle in a data cleaning operation.

# For example, here are some observations I note from reviewing the raw data:

# There is punctuation.
# The text contains uppercase and lowercase.
# There are special characters in the German.
# There are duplicate phrases in English with different translations in German.
# The file is ordered by sentence length with very long sentences toward the end of the file.
# Did you note anything else that could be important?
# Let me know in the comments below.

# A good text cleaning procedure may handle some or all of these observations.

# Data preparation is divided into two subsections:

# Clean Text
# Split Text
# 1. Clean Text
# First, we must load the data in a way that preserves the Unicode German characters. The function below called load_doc() will load the file as a blob of text.

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# Each line contains a single pair of phrases, 
# first English and then German, separated by a tab character.
# We must split the loaded text by line and then by phrase.
# The function to_pairs() below will split the loaded text.

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs

# We are now ready to clean each sentence. The specific cleaning operations we will perform are as follows:

# Remove all non-printable characters.
# Remove all punctuation characters.
# Normalize all Unicode characters to ASCII (e.g. Latin characters).
# Normalize the case to lowercase.
# Remove any remaining tokens that are not alphabetic.
# We will perform these operations on each phrase for each pair in the loaded dataset.

# The clean_pairs() function below implements these operations.

# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

# Finally, now that the data has been cleaned,
# we can save the list of phrase pairs to a file ready for use.

# The function save_clean_data() uses the pickle API to 
# save the list of clean text to file.

# Pulling all of this together, the complete example is listed below.

import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs
 
# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)
 
# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)
 
# load dataset
filename = 'deu.txt'
doc = load_doc(filename)
# split into english-german pairs
pairs = to_pairs(doc)
# clean sentences
clean_pairs = clean_pairs(pairs)
# save clean pairs to file
save_clean_data(clean_pairs, 'english-german.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
