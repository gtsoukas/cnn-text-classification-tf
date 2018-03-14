# Text classification
This code is based on the work of [Danny Britz](https://github.com/dennybritz/cnn-text-classification-tf) and [Yoon Kim](http://arxiv.org/abs/1408.5882).


## Requirements

- Python 3
- Tensorflow 1.5
- Numpy
- Gensim 3.4

## Training

Print parameters:

```bash
python3 ./train.py --help

       USAGE: ./train.py [flags]
flags:

./train.py:
  --[no]allow_soft_placement: Allow soft device placement
    (default: 'true')
  --batch_size: Batch Size
    (default: '64')
    (an integer)
  --character_encoding: Input file encoding.
    (default: 'utf-8')
  --conceptnet_numberbatch: conceptnet-numberbatch file with pre-trained embeddings
  --conceptnet_numberbatch_lang_filter: conceptnet-numberbatch language filter regular expression
    (default: '\\w*')
  --dev_sample_percentage: Percentage of the training data to use for validation
    (default: '0.1')
    (a number)
  --dropout_keep_prob: Dropout keep probability
    (default: '0.5')
    (a number)
  --embedding_dim: Dimensionality of character embedding
    (default: '128')
    (an integer)
  --evaluate_every: Evaluate model on dev set after this many steps
    (default: '100')
    (an integer)
  --fasttext: fastText file with pre-trained embeddings
  --filter_sizes: Comma-separated filter sizes
    (default: '3,4,5')
  --l2_reg_lambda: L2 regularization lambda
    (default: '0.0')
    (a number)
  --[no]log_device_placement: Log placement of ops on devices
    (default: 'false')
  --logs_subfolder: Subfolder within 'runs'-folder
    (default: '')
  --negative_data_file: Data source for the negative data
    (default: './data/rt-polaritydata/rt-polarity.neg')
  --num_epochs: Number of training epochs
    (default: '25')
    (an integer)
  --num_filters: Number of filters per filter size
    (default: '128')
    (an integer)
  --positive_data_file: Data source for the positive data
    (default: './data/rt-polaritydata/rt-polarity.pos')
  --word2vec: Word2vec file with pre-trained embeddings
```

Train:

```bash
./train.py
```

## Evaluating

```bash
./eval.py --checkpoint_dir "./runs/1459637919/best-model/"
```

## References

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)
