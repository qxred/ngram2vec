# ngram2vec
Extension of word2vec, pretrain ngram embeddings


Quite similar to word2vec, except 

1. an additional argument -ngrams, default is 1, i.e. word embedding, -ngrams 2: word + bigram embedding etc. (ngrams < 10)

2. output format: use TABULAR rather than space to split numbers. This is because ngrams can have spaces. 

3. remove argument min-count, threshold is automatically adjusted to fit the memory.

4. remove argument cbow, only support skip-gram.

**Install**

gcc -O3 ngram2vec.c -lpthread -lm

**Quick Start**

Borrow the example from fasttext:

```
$ mkdir data
$ wget -c http://mattmahoney.net/dc/enwik9.zip -P data
$ unzip data/enwik9.zip -d data
$ perl wikifil.pl data/enwik9 > data/fil9
$ mkdir result
$ ./a.out -train data/fil9 -output vec.txt -iter 3 -hs 1 -negative 5 -sample 1e-4 -threads 4 -ngrams 2
```

**Enjoy!!**
