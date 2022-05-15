#!/bin/bash


python3 scripts/average_checkpoints.py \
			--inputs results/mmtimg \
			--num-epoch-checkpoints 20 \
			--output results/mmtimg/model.pt \

