#!/bin/bash

python3 preprocess.py \
  --source-lang en \
  --target-lang de \
  --trainpref data-raw/train.bpe \
  --validpref data-raw/valid.bpe \
  --testpref data-raw/test_2016_flickr.bpe \
  --nwordssrc 17200 \
  --nwordstgt 9800 \
  --workers 12 \
  --destdir data-bin/en-de/test2016 \

  
