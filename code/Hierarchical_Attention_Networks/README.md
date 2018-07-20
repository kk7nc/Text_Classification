# textClassifier

textClassifierHATT.py has the implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). Please see the [my blog](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/) for full detail. Also see [Keras Google group discussion](https://groups.google.com/forum/#!topic/keras-users/IWK9opMFavQ)

textClassifierConv has implemented [Convolutional Neural Networks for Sentence Classification - Yoo Kim](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). Please see the [my blog](https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/) for full detail.

textClassifierRNN has implemented bidirectional LSTM and one level attentional RNN. Please see the [my blog](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/) for full detail.

## update on 6/22/2017 ##
To derive the attention weight which can be useful to identify important words for the classification. Please see my latest update on the post. All you need to do is run a forward pass right before attention layer output. The result is not very promising. I will update the post once I have further result. 
