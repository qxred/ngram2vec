# ngram2vec
Extension of word2vec, pretrain ngram embeddings

Quite similar to word2vec, except 

1. an additional argument -ngrams, default is 1, i.e. word embedding, -ngrams 2: word + bigram embedding etc.

2. output format: use TABULAR rather than space to split numbers. This is because ngrams can have spaces. 

3. remove argument min-count, threshold is automatically adjusted to fit the memory.

4. remove argument cbow, only support skip-gram.
