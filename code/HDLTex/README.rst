|DOI| |travis| |appveyor| |wercker status| |Join the chat at
https://gitter.im/HDLTex| |arXiv| |RG| |Binder| |license| |twitter|

HDLTex: Hierarchical Deep Learning for Text Classification
==========================================================

Refrenced paper : `HDLTex: Hierarchical Deep Learning for Text
Classification <https://arxiv.org/abs/1709.08267>`__

.. image:: /docs/pic/github-logo.png
  :target: https://github.com/kk7nc/HDLTex
  
  
|Pic|

Documentation:
===============

Increasingly large document collections require improved information processing methods for searching, retrieving, and organizing  text. Central to these information processing methods is document classification, which has become an important application for supervised learning. Recently the performance of traditional supervised classifiers has degraded as the number of documents has increased. This is because along with growth in the number of documents has come an increase in the number of categories. This paper approaches this problem differently from current document classification methods that view the problem as multi-class classification. Instead we perform hierarchical classification using an approach we call Hierarchical Deep Learning for Text classification (HDLTex). HDLTex employs stacks of deep learning architectures to provide specialized understanding at each level of the document hierarchy.

Installation
=============

Using pip
----------
.. code:: bash
    
    pip install HDLTex
    
Using git
----------
.. code:: bash

    git clone --recursive https://github.com/kk7nc/HDLTex.git


The primary requirements for this package are Python 3 with Tensorflow.
The requirements.txt file contains a listing of the required Python
packages; to install all requirements, run the following:

.. code:: bash

    pip -r install requirements.txt

Or

.. code:: bash

    pip3  install -r requirements.txt

Or:

.. code:: bash

    conda install --file requirements.txt
        

If the above command does not work, use the following:

.. code:: bash

    sudo -H pip  install -r requirements.txt


Datasets for HDLTex:
=====================

Linke of dataset: |Data|

Web of Science Dataset
`WOS-11967 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

::

        This dataset contains 11,967 documents with 35 categories which include 7 parents categories.
        

Web of Science Dataset
`WOS-46985 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

::

        This dataset contains 46,985 documents with 134 categories which include 7 parents categories.
      

Web of Science Dataset
`WOS-5736 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

::

        This dataset contains 5,736 documents with 11 categories which include 3 parents categories.

Requirements :
----------------
General:

- Python 3.5 or later see `Instruction Documents <https://www.python.org/>`__
- TensorFlow see `Instruction Documents <https://www.tensorflow.org/install/install_linux>`__.
- scikit-learn see `Instruction Documents <http://scikit-learn.org/stable/install.html>`__
- Keras see `Instruction Documents <https://keras.io/>`__
- scipy see `Instruction Documents <https://www.scipy.org/install.html>`__
- GPU

    - CUDA® Toolkit 8.0. For details, see `NVIDIA’s documentation <https://developer.nvidia.com/cuda-toolkit>`__.
    - The `NVIDIA drivers associated with CUDA Toolkit 8.0 <http://www.nvidia.com/Download/index.aspx>`__.
    - cuDNN v6. For details, see `NVIDIA’s documentation <https://developer.nvidia.com/cudnn>`__.
    - GPU card with CUDA Compute Capability 3.0 or higher.
    - The libcupti-dev library,
    - To install this library, issue the following command:

::

        $ sudo apt-get install libcupti-dev
        
        
Feature Extraction:
===================

Global Vectors for Word Representation
(`GLOVE <https://nlp.stanford.edu/projects/glove/>`__)

::

        For CNN and RNN you need to download and linked the folder location to GLOVE
        
        

Error and Comments:
===================

Send an email to kk7nc@virginia.edu

Citation:
=========

.. code:: bash

    @inproceedings{Kowsari2018HDLTex, 
    author={Kowsari, Kamran and Brown, Donald E and Heidarysafa, Mojtaba and Meimandi, Kiana Jafari and Gerber, Matthew S and Barnes, Laura E},
    booktitle={2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
    title={HDLTex: Hierarchical Deep Learning for Text Classification}, 
    year={2017},  
    pages={364-371}, 
    doi={10.1109/ICMLA.2017.0-134},  
    month={Dec}
    }

.. |DOI| image:: http://kowsari.net/HDLTex_DOI.svg?maxAge=2592000
   :target: https://doi.org/10.1109/ICMLA.2017.0-134
.. |travis| image:: https://travis-ci.org/kk7nc/HDLTex.svg?branch=master
   :target: https://travis-ci.org/kk7nc/HDLTex
.. |wercker status| image:: https://app.wercker.com/status/24a123448ba8764b257a1df242146b8e/s/master
   :target: https://app.wercker.com/project/byKey/24a123448ba8764b257a1df242146b8e
.. |Join the chat at https://gitter.im/HDLTex| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/HDLTex/Lobby?source=orgpage
.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/kk7nc/HDLTex?branch=master&svg=true
    :target: https://ci.appveyor.com/project/kk7nc/hdltex
.. |arXiv| image:: https://img.shields.io/badge/arXiv-1709.08267-red.svg?style=flat
   :target: https://arxiv.org/abs/1709.08267
.. |RG| image:: https://img.shields.io/badge/ResearchGate-HDLTex-blue.svg?style=flat
   :target: https://www.researchgate.net/publication/319968747_HDLTex_Hierarchical_Deep_Learning_for_Text_Classification
.. |Binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/kk7nc/HDLTex/master
.. |license| image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592104
   :target: https://github.com/kk7nc/HDLTex/blob/master/LICENSE
.. |Data| image:: https://img.shields.io/badge/DOI-10.17632/9rw3vkcfy4.6-blue.svg?style=flat
   :target: http://dx.doi.org/10.17632/9rw3vkcfy4.6
.. |Pic| image:: http://kowsari.net/____impro/1/onewebmedia/HDLTex.png?etag=W%2F%22c90cd-59c4019b%22&sourceContentType=image%2Fpng&ignoreAspectRatio&resize=821%2B326&extract=0%2B0%2B821%2B325?raw=false
   :alt: HDLTex as both Hierarchy lavel are DNN
.. |twitter| image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social
   :target: https://twitter.com/intent/tweet?text=HDLTex:%20Hierarchical%20Deep%20Learning%20for%20Text%20Classification%0aGitHub:&url=https://github.com/kk7nc/HDLTex&hashtags=DeepLearning,Text_Classification,classification,MachineLearning,deep_neural_networks
       
