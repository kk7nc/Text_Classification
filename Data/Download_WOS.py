Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @kk7nc Sign out
14
72 25 kk7nc/RMDL
 Code  Issues 1  Pull requests 0  Projects 0  Wiki  Insights  Settings
RMDL/RMDL/Download/Download_WOS.py
97366d9  on Jun 4
@kk7nc kk7nc bug fixes
     
64 lines (44 sloc)  1.88 KB
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
RMDL: Random Multimodel Deep Learning for Classification
 * Copyright (C) 2018  Kamran Kowsari <kk7nc@virginia.edu>
 * Last Update: 04/25/2018
 * This file is part of  RMDL project, University of Virginia.
 * Free to use, change, share and distribute source code of RMDL
 * Refrenced paper : RMDL: Random Multimodel Deep Learning for Classification
 * Refrenced paper : An Improvement of Data Classification using Random Multimodel Deep Learning (RMDL)
 * Comments and Error: email: kk7nc@virginia.edu
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import print_function

import os, sys, tarfile
import numpy as np

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape


# path to the directory with the data
DATA_DIR = '.\data_WOS'

# url of the binary data
DATA_URL = 'http://kowsari.net/WebOfScience.tar.gz'


# path to the binary train file with image data


def download_and_extract():
    """
    Download and extract the WOS datasets
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)


    path = os.path.abspath(dest_directory)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)

        print('Downloaded', filename)

        tarfile.open(filepath, 'r').extractall(dest_directory)
    return path
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About
Press h to open a hovercard with more details.
