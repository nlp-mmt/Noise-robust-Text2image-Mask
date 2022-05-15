#!/bin/bash

python3 generate.py  data-bin/en-de/test2016 \
				--path results/mmtimg/model.pt \
				--source-lang en --target-lang de \
				--beam 5 \
				--num-workers 12 \
				--batch-size 128 \
				--results-path results/mmtimg/test2016 \
				--remove-bpe \
#				--fp16 \