# Brief summary of experiment of multi-context HAN
#### Multi-context number = 10:
<img src="https://github.com/WenjianDong/deep_learning_NLP/blob/master/lm01_new.png" alt="" width="400"/>
<img src="https://github.com/WenjianDong/deep_learning_NLP/blob/master/lm02_new.png" alt="" width="400"/>
<img src="https://github.com/WenjianDong/deep_learning_NLP/blob/master/lm10_new.png" alt="" width="400"/>
Compared to what I drew for the defence, the loss these figures is direct loss without summing along samples in a batch. It demonstrates the same phenomenon as the one during the defence: the when lambda is 0.1 or 0.2 the regularization term doesn't influence much on the results, but when lambda is as large as 1.0, the result is not stable. 
<br>
In addition, I did an experiment with multi-context number is 3 and lambda=0.2. The final accuracy is about 63.0%.
<img src="https://github.com/WenjianDong/deep_learning_NLP/blob/master/multi3-lm02.png" alt="" width="550"/>


# Deep Learning architectures for NLP

This repository contains Keras implementations of the architectures listed below. For a quick theoretical intro about Deep Learning for NLP, I encourage you to have a look at my [notes](https://arxiv.org/pdf/1808.09772.pdf).

## Hierarchical Attention Network for Document Classification
A **RAM-friendly** implementation of the model introduced by [Yang et al. (2016)](http://www.aclweb.org/anthology/N16-1174), with step-by-step explanations and links to relevant resources: https://github.com/Tixierae/deep_learning_NLP/blob/master/HAN/HAN_final.ipynb

<img src="https://raw.githubusercontent.com/Tixierae/deep_learning_NLP/master/HAN/han_architecture_illustration_small.bmp" alt="" width="550"/>

In my experiments on the **Amazon review dataset** (3,650,000 documents, 5 classes), I reach **62.6%** accuracy after 8 epochs, and **63.6%** accuracy (the accuracy reported in the paper) after 42 epochs. Each epoch takes about 20 mins on my TitanX GPU. I deployed the model as a [web app](https://safetyapp.shinyapps.io/DNLPvis/). As shown in the image below, you can paste your own review and visualize how the model pays attention to words and sentences.

<a href="https://safetyapp.shinyapps.io/DNLPvis/" target="_blank">
<img src="https://raw.githubusercontent.com/Tixierae/deep_learning_NLP/master/HAN/dnlpvis_app_illustration.bmp" alt="" width="450"/></a>

### Concepts covered
The notebook makes use of the following concepts:

- **batch training**. Batches are loaded from disk and passed to the model one by one with a generator. This way, it's possible to train on datasets that are too big to fit on RAM. 
- **bucketing**. To have batches that are as dense as possible and make the most of each tensor product, the batches contain documents of similar sizes.
- **cyclical learning rate and cyclical momentum schedules**, as in [Smith (2017)](https://arxiv.org/pdf/1506.01186.pdf) and [Smith (2018)](https://arxiv.org/pdf/1803.09820.pdf). The cyclical learning rate schedule is a new, promising approach to optimization in which the learning rate increases and decreases in a pre-defined interval rather than keeping decreasing. It worked better than Adam and SGD alone for me<sup>1</sup>.
- **self-attention** (aka inner attention). We use the formulation of the original paper.
- **bidirectional RNN**
- **Gated Recurrent Unit (GRU)** 

<sup>1</sup>There is more and more evidence that adaptive optimizers like Adam, Adagrad, etc. converge faster but generalize poorly compared to SGD-based approaches. For example: [Wilson et al. (2018)](https://arxiv.org/pdf/1705.08292.pdf), this [blogpost]( https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/). Traditional SGD is very slow, but a cyclical learning rate schedule can bring a significant speedup, and even sometimes allow to reach better performance.

## 1D Convolutional Neural Network for short text classification
An implementation of [(Kim 2014)'s](https://arxiv.org/abs/1408.5882) 1D Convolutional Neural Network for short text classification: https://github.com/Tixierae/deep_learning_NLP/blob/master/cnn_imdb.ipynb 

<img src="https://github.com/Tixierae/deep_learning_NLP/raw/master/cnn_illustration.png" alt="Drawing" style="width: 250px;"/>

## 2D CNN for image classification
Agreed, this is not for NLP. But an implementation can be found here https://github.com/Tixierae/deep_learning_NLP/blob/master/mnist_cnn.py. I reach 99.45% accuracy on MNIST with it.

## Cite
If you use some of the code in this repository, please cite
```
@article{tixier2018notes,
  title={Notes on Deep Learning for NLP},
  author={Tixier, Antoine J.-P.},
  journal={arXiv preprint arXiv:1808.09772},
  year={2018}
}
```
