
################################################
Text Classification Algorithm: A Brief Overview
################################################

##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

============
Introduction
============


|line|

====================================
Text and Document Feature Extraction
====================================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Text Cleaning and Pre-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-------------
Tokenization
-------------

Tokenization is a part of pre-process to break a stream of text up into words, phrases, symbols, or other meaningful elements called tokens.  The main goal of this step is the exploration of the words in a sentence. In text mining beside of text classification, it;'s necessitate a parser which processes the tokenization of the documents; for example:

sentence:

.. code::

  After sleeping for four hours, he decided to sleep for another four


In this case, the tokens are as follows:

.. code::

    {'After', 'sleeping', 'for', 'four', 'hours', 'he', 'decided', 'to', 'sleep', 'for', 'another', 'four'}


Here is python code for Tokenization:

.. code:: python

  from nltk.tokenize import word_tokenize
  text = "After sleeping for four hours, he decided to sleep for another four"
  tokens = word_tokenize(text)
  print(tokens)

-----------
Stop words
-----------


Text and document classification over social media such as Twitter, Facebook, and so on is usually affected by the noisy nature (abbreviations, irregular forms) of these data points.

Here is an exmple from  `geeksforgeeks <https://www.geeksforgeeks.org/removing-stop-words-nltk-python/>`__

.. code:: python

  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize

  example_sent = "This is a sample sentence, showing off the stop words filtration."

  stop_words = set(stopwords.words('english'))

  word_tokens = word_tokenize(example_sent)

  filtered_sentence = [w for w in word_tokens if not w in stop_words]

  filtered_sentence = []

  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)

  print(word_tokens)
  print(filtered_sentence)



Output:

.. code::

  ['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 
  'off', 'the', 'stop', 'words', 'filtration', '.']
  ['This', 'sample', 'sentence', ',', 'showing', 'stop',
  'words', 'filtration', '.']


---------------
Capitalization
---------------

Text and document data points have a diversity of capitalization to became a sentence; substantially, several sentences together create a document. The most common approach of capitalization method could be to reduce everything to lower case. This technique makes all words in text and document in same space, but it is caused to a significant problem for meaning of some words such as "US" to "us" which first one represent the country of United States of America and second one is pronouns word; thus, for solving this problem, we could use slang and abbreviation converters.

.. code:: python

  text = "The United States of America (USA) or America, is a federal republic composed of 50 states"
  print(text)
  print(text.lower())

Output:

.. code:: python

  "The United States of America (USA) or America, is a federal republic composed of 50 states"
  "the united states of america (usa) or america, is a federal republic composed of 50 states"

-----------------------
Slang and Abbreviation
-----------------------

Slang and Abbreviation is another problem as pre-processing step for cleaning text datasets. An abbreviation  is a shortened form of a word or phrase which contain mostly first letters form the words such as SVM stand for  Support Vector Machine. Slang is a version of language of an informal talk or text that has different meaning such as "lost the plot", it essentially means that they've gone mad. The common method for dealing with these words is convert them to formal language.

---------------
Noise Removal
---------------


The other issue of text cleaning as pre-processing step is noise removal which most of text and document datasets contains many unnecessary characters such as punctuation, special character. It's important to know the punctuation is critical for us to understand the meaning of the sentence, but it could have effect for classification algorithms.


Here is simple code to remove standard noise from text:


.. code:: python

  def text_cleaner(text):
      rules = [
          {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
          {r'\s+': u' '},  # replace consecutive spaces
          {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
          {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
          {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
          {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
          {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
          {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
          {r'^\s+': u''}  # remove spaces at the beginning
      ]
      for rule in rules:
      for (k, v) in rule.items():
          regex = re.compile(k)
          text = regex.sub(v, text)
      text = text.rstrip()
      return text.lower()
    


-------------------
Spelling Correction
-------------------


One of the optional part of the pre-processing step is spelling correction which is happened in texts and documents. Many algorithm, techniques, and methods have been addressed this problem in NLP. Many techniques and methods are available for researchers such as hashing-based and context-sensitive spelling correction techniques, or  spelling correction using trie and damerau-levenshtein distance bigram.


.. code:: python

  from autocorrect import spell

  print spell('caaaar')
  print spell(u'mussage')
  print spell(u'survice')
  print spell(u'hte')

Result:

.. code::

    caesar
    message
    service
    the


------------
Stemming
------------


Text Stemming is modifying to obtain variant word forms using different linguistic processes such as affixation (addition of affixes). For example, the stem of the word "studying" is "study", to which -ing.


Here is an example of Stemming from `NLTK <https://pythonprogramming.net/stemming-nltk-tutorial/>`__

.. code:: python

    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize

    ps = PorterStemmer()

    example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
    
    for w in example_words:
    print(ps.stem(w))


Result:

.. code::

  python
  python
  python
  python
  pythonli

-------------
Lemmatization
-------------


Text lemmatization is process in NLP to replaces the suffix of a word with a different one or removes the suffix of a word completely to get the basic word form (lemma).


.. code:: python

  from nltk.stem import WordNetLemmatizer

  lemmatizer = WordNetLemmatizer()

  print(lemmatizer.lemmatize("cats"))

~~~~~~~~~~~~~~
Word Embedding
~~~~~~~~~~~~~~

--------
Word2Vec
--------

----------------------------------------------
Global Vectors for Word Representation (GloVe)
----------------------------------------------

--------
FastText
--------

~~~~~~~~~~~~~~
Weighted Words
~~~~~~~~~~~~~~


--------------
Term frequency
--------------


-----------------------------------------
Term Frequency-Inverse Document Frequency
-----------------------------------------
The mathematical representation of weight of a term in a document by Tf-idf is given:

.. image:: docs/eq/tf-idf.gif
   :width: 10px
   
Where N is number of documents and df(t) is the number of documents containing the term t in the corpus. The first part would improve recall and the later would improve the precision of the word embedding. Although tf-idf tries to overcome the problem of common terms in document, it still suffers from some other descriptive limitations. Namely, tf-idf cannot account for the similarity between words in the document since each word is presented as an index. In the recent years, with development of more complex models such as neural nets, new methods has been presented that can incorporate concepts such as similarity of words and part of speech tagging. This work uses, word2vec and Glove, two of the most common methods that have been successfully used for deep learning techniques.


.. code:: python

  from sklearn.feature_extraction.text import TfidfTransformer
  def loadData(X_train, X_test,MAX_NB_WORDS=75000):
      vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
      X_train = vectorizer_x.fit_transform(X_train).toarray()
      X_test = vectorizer_x.transform(X_test).toarray()
      print("tf-idf with",str(np.array(X_train).shape[1]),"features")
      return (X_train,X_test)

========================
Dimensionality Reduction
========================


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Principal Component Analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Linear Discriminant Analysis (LDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-negative Matrix Factorization (NMF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~
Random Projection
~~~~~~~~~~~~~~~~~

~~~~~~~~~~~
Autoencoder
~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T-distributed Stochastic Neighbor Embedding (T-SNE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



===============================
Text Classification Techniques
===============================



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rocchio classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boosting and Bagging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

---------
Boosting
---------

-------
Bagging
-------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Naive Bayes Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K-nearest Neighbor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support Vector Machine~(SVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decision Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Random Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conditional Random Field (CRF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Deep Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-----------------------------------------
Deep Neural Networks
-----------------------------------------


-----------------------------------------
Recurrent Neural Networks (RNN)
-----------------------------------------



-----------------------------------------
Convolutional Neural Networks (CNN)
-----------------------------------------



-----------------------------------------
Deep Belief Network (DBN)
-----------------------------------------



-----------------------------------------
Hierarchical Attention Networks
-----------------------------------------


---------------------------------------------
Recurrent Convolutional Neural Networks (RCNN)
---------------------------------------------


-----------------------------------------
Random Multimodel Deep Learning (RMDL)
-----------------------------------------


Referenced paper : `RMDL: Random Multimodel Deep Learning for
Classification <https://www.researchgate.net/publication/324922651_RMDL_Random_Multimodel_Deep_Learning_for_Classification>`__


A new ensemble, deep learning approach for classification. Deep
learning models have achieved state-of-the-art results across many domains.
RMDL solves the problem of finding the best deep learning structure
and architecture while simultaneously improving robustness and accuracy
through ensembles of deep learning architectures. RDML can accept
asinput a variety data to include text, video, images, and symbolic.


|RMDL|

Random Multimodel Deep Learning (RDML) architecture for classification.
RMDL includes 3 Random models, oneDNN classifier at left, one Deep CNN
classifier at middle, and one Deep RNN classifier at right (each unit could be LSTMor GRU).


Installation

There are pip and git for RMDL installation:

Using pip


.. code:: python

        pip install RMDL

Using git

.. code:: bash

    git clone --recursive https://github.com/kk7nc/RMDL.git

The primary requirements for this package are Python 3 with Tensorflow. The requirements.txt file
contains a listing of the required Python packages; to install all requirements, run the following:

.. code:: bash

    pip -r install requirements.txt

Or

.. code:: bash

    pip3  install -r requirements.txt

Or:

.. code:: bash

    conda install --file requirements.txt

Documentation:


The exponential growth in the number of complex datasets every year requires  more enhancement in
machine learning methods to provide  robust and accurate data classification. Lately, deep learning
approaches have been achieved surpassing results in comparison to previous machine learning algorithms
on tasks such as image classification, natural language processing, face recognition, and etc. The
success of these deep learning algorithms relys on their capacity to model complex and non-linear
relationships within data. However, finding the suitable structure for these models has been a challenge
for researchers. This paper introduces Random Multimodel Deep Learning (RMDL): a new ensemble, deep learning
approach for classification.  RMDL solves the problem of finding the best deep learning structure and
architecture while simultaneously improving robustness and accuracy through ensembles of deep
learning architectures. In short, RMDL trains multiple models of Deep Neural Network (DNN),
Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) in parallel and combines
their results to produce better result of any of those models individually. To create these models,
each deep learning model has been constructed in a random fashion regarding the number of layers and
nodes in their neural network structure. The resulting RDML model can be used for various domains such
as text, video, images, and symbolic. In this Project, we describe RMDL model in depth and show the results
for image and text classification as well as face recognition. For image classification, we compared our
model with some of the available baselines using MNIST and CIFAR-10 datasets. Similarly, we used four
datasets namely, WOS, Reuters, IMDB, and 20newsgroup and compared our results with available baselines.
Web of Science (WOS) has been collected  by authors and consists of three sets~(small, medium and large set).
Lastly, we used ORL dataset to compare the performance of our approach with other face recognition methods.
These test results show that RDML model consistently outperform standard methods over a broad range of
data types and classification problems.

--------------------------------------------
Hierarchical Deep Learning for Text (HDLTex)
--------------------------------------------

Refrenced paper : `HDLTex: Hierarchical Deep Learning for Text
Classification <https://arxiv.org/abs/1709.08267>`__


|HDLTex|

Documentation:

Increasingly large document collections require improved information processing methods for searching, retrieving, and organizing  text. Central to these information processing methods is document classification, which has become an important application for supervised learning. Recently the performance of traditional supervised classifiers has degraded as the number of documents has increased. This is because along with growth in the number of documents has come an increase in the number of categories. This paper approaches this problem differently from current document classification methods that view the problem as multi-class classification. Instead we perform hierarchical classification using an approach we call Hierarchical Deep Learning for Text classification (HDLTex). HDLTex employs stacks of deep learning architectures to provide specialized understanding at each level of the document hierarchy.



------------------------------------------------
Semi-supervised learning for Text classification
------------------------------------------------




==========
Evaluation
==========

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
F1 Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Matthew correlation coefficient (MCC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Receiver operating characteristics (ROC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~
Area under curve~(AUC)
~~~~~~~~~~~~~~~~~~~~~~~


==========================
Text and Document Datasets
==========================

~~~~~
IMDB
~~~~~

- `IMDB Dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`__

  * This dataset contains 50,000 documents with 2 categories.

~~~~~~~~~~~~~
Reters-21578
~~~~~~~~~~~~~

- `Reters-21578 Dataset <https://keras.io/datasets/>`__

  * This dataset contains 21,578 documents with 90 categories.

~~~~~~~~~~~~~
20Newsgroups
~~~~~~~~~~~~~

- `20Newsgroups Dataset <https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups>`__

  * This dataset contains 20,000 documents with 20 categories.


~~~~~~~~~~~~~~~~~~~~~~
Web of Science Dataset
~~~~~~~~~~~~~~~~~~~~~~

-  Web of Science Dataset (DOI:
   `10.17632/9rw3vkcfy4.2 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__)

   -  Web of Science Dataset
      `WOS-11967 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 11,967 documents with 35 categories which
         include 7 parents categories.

   -  Web of Science Dataset
      `WOS-46985 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 46,985 documents with 134 categories
         which include 7 parents categories.

   -  Web of Science Dataset
      `WOS-5736 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

      -  This dataset contains 5,736 documents with 11 categories which
         include 3 parents categories.
         
===========
Application
===========

~~~~~~~~~~~~~~~~~~~~~~~~
Information Retrieval
~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~
Information Filtering
~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~~~~~~~~
Sentiment Analysis
~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~~~~~~~~
Healthcare
~~~~~~~~~~~~~~~~~~~~~~~~



~~~~~~~~~~~~~~~~~~~~~~~~
Human Behavior
~~~~~~~~~~~~~~~~~~~~~~~~




~~~~~~~~~~~~~~~~~~~~~~~~
Knowledge Management
~~~~~~~~~~~~~~~~~~~~~~~~




~~~~~~~~~~~~~~~~~~~~~~~~
Safety and Security
~~~~~~~~~~~~~~~~~~~~~~~~


~~~~~~~~~~~~~~~~~~~~~~~~
Recommender Systems
~~~~~~~~~~~~~~~~~~~~~~~~



=========
Citations
=========
.. code::

    @inproceedings{Kowsari2018Text_Classification,
    title={Text Classification Algorithm: A Brief Overview},
    author={Kowsari, Kamran and Jafari Meimandi, Kiana and Heidarysafa, Mojtaba and Gerber Matthew S. and  Barnes, Laura E. and Brown, Donald E.},
    booktitle={},
    year={2018},
    DOI={https://doi.org/},
    organization={IEEE}
    }

.. |RMDL| image:: http://kowsari.net/onewebmedia/RMDL.jpg
.. |line| image:: docs/pic/line.png
.. |HDLTex| image:: http://kowsari.net/____impro/1/onewebmedia/HDLTex.png?etag=W%2F%22c90cd-59c4019b%22&sourceContentType=image%2Fpng&ignoreAspectRatio&resize=821%2B326&extract=0%2B0%2B821%2B325?raw=false
