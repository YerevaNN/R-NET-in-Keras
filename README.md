# R-NET implementation in Keras

This repository is an attempt to reproduce the results presented in the [technical report by Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). The report describes a complex neural network called [R-NET](https://www.microsoft.com/en-us/research/publication/mrc/) designed for question answering.

R-NET is currently (July 2017) the best model on Stanford QA database: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). SQuAD dataset uses two performance metrics, exact match (EM) and F1-score (F1). Human performance is estimated to be EM=82.3% and F1=91.2% on the test set. 

The report describes two versions of R-NET:
1. The first one is called `R-NET (Wang et al., 2017)` (which refers to a paper which not yet available online) and reaches EM=71.3% and F1=79.7% on the test set. It consists of input encoders, a modified version of Match-LSTM, self-matching attention layer (the main contribution of the paper) and a pointer network. 
2. The second version called `R-NET (March 2017)` has one additional BiGRU between the self-matching attention layer and the pointer network and reaches EM=72.3% and F1=80.7%.

The current best single-model on SQuAD leaderboard has a higher score, which means R-NET development continued after March 2017. Ensemble models reach higher scores.

This repository contains an implementation of the first version, but we cannot yet reproduce the reported results. The best performance we got so far was EM=56.82% and F1=66.68% on the dev set. We are aware of a few differences between our implementation and the network described in the paper:

1. We do not use character-level embedding at the input.
2. The first formula in (11) of the [report](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) contains a strange summand W_v^Q V_r^Q. Both tensors are trainable and are not used anywhere else in the network. We have replaced this product with a single trainable vector.
3. The size of the hidden layer should 75 according to the report, but we get better results with a lower number. Overfitting is huge with 75 neurons.

We are not sure whether we applied dropout correctly. Also there is nothing about weight initialization in the report. On the other hand we can't rule out that we have bugs in our code.

## Instructions

1. We need to parse and split the data
```sh
    python parse_data.py data/train-v1.1.json --train_ratio 0.9 --outfile data/train_parsed.json --outfile_valid data/valid_parsed.json
    python parse_data.py data/train-v1.1.json --outfile data/train_parsed.json
```

2. Preprocess the data
```sh
    python preprocessing.py data/train_parsed.json --outfile data/train_data_str.pkl --include_str
    python preprocessing.py data/valid_parsed.json --outfile data/valid_data_str.pkl --include_str
    python preprocessing.py data/dev_parsed.json --outfile data/dev_data_str.pkl --include_str
```

3. Train the model
```sh
    python train.py --hdim 45 --batch_size 50 --nb_epochs 50 --optimizer adadelta --lr 1 --dropout 0.2 --char_level_embeddings --train_data data/train_data_str.pkl --valid_data data/valid_data_str.pkl
```

4. Predict on dev/test set samples
```sh
    python predict.py --batch_size 100 --dev_data data/dev_data_str.pkl models/31-t3.05458271443-v3.27696280528.model prediction.json
```
