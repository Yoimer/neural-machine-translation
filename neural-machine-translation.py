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
