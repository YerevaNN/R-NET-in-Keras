# R-NET implementation in Keras

This repository is an attempt to reproduce the results presented in the [technical report by Microsoft Research Asia](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf). The report describes a complex neural network called [R-NET](https://www.microsoft.com/en-us/research/publication/mrc/) designed for question answering.

R-NET is currently (July 2017) the best model on Stanford QA database: [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). SQuAD dataset uses two performance metrics, exact match (EM) and F1-score (F1). Human performance is estimated to be EM=82.3% and F1=91.2%. 

The report describes two versions of R-NET:
1. One is marked as `R-NET (Wang et al., 2017)` (which refers to a paper which not yet available online) and reaches EM=71.3% and F1=79.7%. It consists of input encoders, a modified version of Match-LSTM, self-matching attention layer (the main contribution of the paper) and a pointer network. 
2. The second version `R-NET (March 2017)` has one additional BiGRU between the self-matching attention layer and the pointer network and reaches EM=72.3% and F1=80.7%.

The current best single-model on SQuAD leaderboard has a higher score, which means R-NET development continued after March 2017. Ensemble models reach higher scores.

This repository contains an implementation of the first version, but we cannot yet reproduce the reported results. The best performance we got so far was EM=54.21% and F1=65.26%. We are aware of two differences between our implementation and the network described in the paper:

1. We do not use character-level embedding at the input.
2. The first formula in (11) of the [report](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf) contains a strange summand W_v^Q V_r^Q. Both tensors are trainable and are not used anywhere else in the network. We have replaced this product with a single trainable vector.
