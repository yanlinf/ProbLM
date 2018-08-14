#!/bin/bash

export PYTHONHASHSEED=0

python multiprocess_train.py corpus/enwiki.txt --hash_size 262144 -ns 3 --verbose --output models/enwiki_ns3