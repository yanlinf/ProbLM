# ProbLM
probabilistic counting for language modeling.

## Description

A language model computes the probability of a sentence:
$$
P(w_1, \dots , w_m)=\prod_{i=1}^m P(w_i \mid w_1, \dots , w_{i-1}) \approx \prod_{i=1}^m P(w_i \mid w_{i-(n-1)}, \dots , w_{i-1})
$$
where $P(w_i \mid w_{i-(n-1)}, \dots , w_{i-1})$ can be estimated by counting the occurance of n-grams:
$$
P(w_i \mid w_{i-(n-1)}, \dots , w_{i-1})\approx \frac{\text{count}( w_{i-(n-1)}, \dots , w_{i-1}, w_i)}{\text{count}(w_{i-(n-1)}, \dots , w_{i-1})}
$$
and with smoothing (so that model will not output zero probability for an unseen n-gram):
$$
P(w_i \mid w_{i-(n-1)}, \dots , w_{i-1})\approx \frac{\text{count}( w_{i-(n-1)}, \dots , w_{i-1}, w_i)+1}{\text{count}(w_{i-(n-1)}, \dots , w_{i-1})+|V|}
$$
where $|V|$ is the vocabulary size.



Traditional  method maintain a huge table to store the counts of every n-gram, which is memory-inefficient, since there are $|V|^n$ possible n-grams and $|V|$ is already large. Our implementation use memory-efficient **CountMinSketch** algorithm to estimate n-gram counts and **HyperLogLog** to estimate vocabulary size.

## Requirements

1. python3
2. numpy

## Usage

### 0. Prepare the corpus (optional)

1.  Download a English wikipedia corpus:

   ```bash
   >>> mkdir corpus
   >>> wget https://dumps.wikimedia.org/enwiki/20180801/enwiki-20180801-pages-articles-multistream.xml.bz2 -P corpus/
   ```

2.  Convert the corpus to plain text (require gensim):

   ```bash
   >>> python make_corpus.py corpus/enwiki-20180801-pages-articles-multistream.xml.bz2 corpus/enwiki.txt
   ```

### 1. Train the model

```bash
>>> ./train.sh

2018-08-15 00:00:11,798: INFO: processed 100 lines
2018-08-15 00:00:27,022: INFO: processed 200 lines
2018-08-15 00:00:39,387: INFO: processed 300 lines
2018-08-15 00:00:51,641: INFO: processed 400 lines
2018-08-15 00:01:00,908: INFO: processed 500 lines
2018-08-15 00:01:07,962: INFO: model saved to models/enwiki_ns3
```

### 2. Evaluate the model

```bash
>>> ./human_eval.sh

INFO:root:type = 'count_min_sketch'
INFO:root:counter = <frequency_estimation.CountMinSketch object at 0x000001A3DF80C8D0>
INFO:root:vocab_size = 82346
INFO:root:hash_size = 1048576
INFO:root:hash_num = 32
INFO:root:ngram_size = 3
Enter a sentence (EXIT to break):

>>> it is important

INFO:root:('<BOS>', '<BOS>')     count = 500
INFO:root:('<BOS>', '<BOS>', 'it')       count = 2
INFO:root:('<BOS>', 'it')        count = 2
INFO:root:('<BOS>', 'it', 'is')  count = 2
INFO:root:('it', 'is')   count = 1817
INFO:root:('it', 'is', 'important')      count = 10
INFO:root:('is', 'important')    count = 33
INFO:root:('is', 'important', '<EOS>')   count = 0

------------------------------------------
Probability: 0.0000000000000000020930175548966076439696
```

