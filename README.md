# R-NET implementation in Keras

This repository is an attempt to reproduce the results presented in the [technical report by Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). The report describes a complex neural network called [R-NET](https://www.microsoft.com/en-us/research/publication/mrc/) designed for question answering.

**[This blogpost](http://yerevann.github.io/2017/08/25/challenges-of-reproducing-r-net-neural-network-using-keras/) describes the details.**

R-NET is currently (August 25, 2017) the best single model on the Stanford QA database: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). SQuAD dataset uses two performance metrics, exact match (EM) and F1-score (F1). Human performance is estimated to be EM=82.3% and F1=91.2% on the test set. 

The report describes two versions of R-NET:
1. The first one is called `R-NET (Wang et al., 2017)` (which refers to a paper which not yet available online) and reaches EM=71.3% and F1=79.7% on the test set. It consists of input encoders, a modified version of Match-LSTM, self-matching attention layer (the main contribution of the paper) and a pointer network. 
2. The second version called `R-NET (March 2017)` has one additional BiGRU between the self-matching attention layer and the pointer network and reaches EM=72.3% and F1=80.7%.

The current best single-model on SQuAD leaderboard has a higher score, which means R-NET development continued after March 2017. Ensemble models reach higher scores.

This repository contains an implementation of the first version, but we cannot yet reproduce the reported results. The best performance we got so far was EM=57.52% and F1=67.42% on the dev set. We are aware of a few differences between our implementation and the network described in the paper:

1. The first formula in (11) of the [report](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) contains a strange summand W_v^Q V_r^Q. Both tensors are trainable and are not used anywhere else in the network. We have replaced this product with a single trainable vector.
2. The size of the hidden layer should 75 according to the report, but we get better results with a lower number. Overfitting is huge with 75 neurons.
3. We are not sure whether we applied dropout correctly. 
4. There is nothing about weight initialization or batch generation in the report.
5. Question-aware passage representation generation (probably) should be done by a bidirectional GRU.

On the other hand we can't rule out that we have bugs in our code.

## Instructions

1. We need to download Glove data and convert it to word2vec format
```sh
python word2vec_downloader.py --word2vec_path data/word2vec_from_glove_300.vec
```

2. Parse and split the data
```sh
python parse_data.py data/train-v1.1.json --train_ratio 0.9 --outfile data/train_parsed.json --outfile_valid data/valid_parsed.json
python parse_data.py data/dev-v1.1.json --outfile data/dev_parsed.json
```

3. Preprocess the data
```sh
python preprocessing.py data/train_parsed.json --outfile data/train_data_str.pkl --include_str
python preprocessing.py data/valid_parsed.json --outfile data/valid_data_str.pkl --include_str
python preprocessing.py data/dev_parsed.json --outfile data/dev_data_str.pkl --include_str
```

4. Train the model
```sh
python train.py --hdim 45 --batch_size 50 --nb_epochs 50 --optimizer adadelta --lr 1 --dropout 0.2 --char_level_embeddings --train_data data/train_data_str.pkl --valid_data data/valid_data_str.pkl
```

5. Predict on dev/test set samples
```sh
python predict.py --batch_size 100 --dev_data data/dev_data_str.pkl models/31-t3.05458271443-v3.27696280528.model prediction.json
```

Our best model can be downloaded from Release v0.1: https://github.com/YerevaNN/R-NET-in-Keras/releases/download/v0.1/31-t3.05458271443-v3.27696280528.model
