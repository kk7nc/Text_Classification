
################################################
Text Classification Algorithm: A Brief Overview
################################################

##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4


IMDB
-----

-  This dataset contains 50,000 documents with 2 categories.

Import Packages
~~~~~~~~~~~~~~~

.. code:: python

        import sys
        import os
        from RMDL import text_feature_extraction as txt
        from keras.datasets import imdb
        import numpy as np
        from RMDL import RMDL_Text as RMDL

Load Data
~~~~~~~~~

.. code:: python

        print("Load IMDB dataset....")
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
        print(len(X_train))
        print(y_test)
        word_index = imdb.get_word_index()
        index_word = {v: k for k, v in word_index.items()}
        X_train = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_train]
        X_test = [txt.text_cleaner(' '.join(index_word.get(w) for w in x)) for x in X_test]
        X_train = np.array(X_train)
        X_train = np.array(X_train).ravel()
        print(X_train.shape)
        X_test = np.array(X_test)
        X_test = np.array(X_test).ravel()
        
        
Web Of Science
--------------

-  Linke of dataset:  |Data|

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

Import Packages
~~~~~~~~~~~~~~~

.. code:: python

        from RMDL import text_feature_extraction as txt
        from sklearn.model_selection import train_test_split
        from RMDL.Download import Download_WOS as WOS
        import numpy as np
        from RMDL import RMDL_Text as RMDL

Load Data
~~~~~~~~~
.. code:: python

        path_WOS = WOS.download_and_extract()
        fname = os.path.join(path_WOS,"WebOfScience/WOS11967/X.txt")
        fnamek = os.path.join(path_WOS,"WebOfScience/WOS11967/Y.txt")
        with open(fname, encoding="utf-8") as f:
            content = f.readlines()
            content = [txt.text_cleaner(x) for x in content]
        with open(fnamek) as fk:
            contentk = fk.readlines()
        contentk = [x.strip() for x in contentk]
        Label = np.matrix(contentk, dtype=int)
        Label = np.transpose(Label)
        np.random.seed(7)
        print(Label.shape)
        X_train, X_test, y_train, y_test = train_test_split(content, Label, test_size=0.2, random_state=4)
        
        
        
Reuters-21578
-------------

- This dataset contains 21,578 documents with 90 categories.

Import Packages
~~~~~~~~~~~~~~~

.. code:: python

         import sys
         import os
         import nltk
         nltk.download("reuters")
         from nltk.corpus import reuters
         from sklearn.preprocessing import MultiLabelBinarizer
         import numpy as np
         from RMDL import RMDL_Text as RMDL

Load Data
~~~~~~~~~
.. code:: python

         documents = reuters.fileids()

         train_docs_id = list(filter(lambda doc: doc.startswith("train"),
                                   documents))
         test_docs_id = list(filter(lambda doc: doc.startswith("test"),
                                  documents))
         X_train = [(reuters.raw(doc_id)) for doc_id in train_docs_id]
         X_test = [(reuters.raw(doc_id)) for doc_id in test_docs_id]
         mlb = MultiLabelBinarizer()
         y_train = mlb.fit_transform([reuters.categories(doc_id)
                                    for doc_id in train_docs_id])
         y_test = mlb.transform([reuters.categories(doc_id)
                               for doc_id in test_docs_id])
         y_train = np.argmax(y_train, axis=1)
         y_test = np.argmax(y_test, axis=1)




    
