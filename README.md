# MultimodelMixed-MMT
 Leveraging Image and Text with MultimodelMixup for Multi-model Neural Machine Translation
## Requirements
ubuntu  
cuda==10.2  
python==3.6  
torch==1.5.1  
## dataset
txt dataWe employ the data set [Multi30K data set](http://www.statmt.org/wmt18/multimodal-task.html), then use [BPE](https://github.com/rsennrich/subword-nmt) to preprocess the raw data(dataset/data/task1/tok/). Image features are extracted through the pre-trained Resnet-101.  
##### BPE (learn_joint_bpe_and_vocab.py and apply_bpe.py)
English, German, French use BPE participle separately.   
-s 10000 \
--vocabulary-threshold 1 \
## MultimodelMixed-MMT Quickstart
### Step 1: preprocess.py  
  --source-lang $SRC_LANG \
  --target-lang $TGT_LANG \
  --trainpref $TMP_DIR/train.bpe \
  --validpref $TMP_DIR/val.bpe \
  --testpref $TMP_DIR/test_2016_flickr.bpe \
  --nwordssrc 17200 \
  --nwordstgt 9800 \
  --workers 12 \
  --destdir $DATA_DIR   
##### Then add the pre-trained Resnet-101 image feature to $DATA_DIR 
### Step 2: train.py  
  $DATA_DIR  
  --arch transformer_iwslt_de_en  \
  --share-decoder-input-output-embed \
  --clip-norm 0 --optimizer adam --lr 0.001 \
  --source-lang $SRC_LANG --target-lang $TGT_LANG --max-tokens 1536 --no-progress-bar \
  --log-interval 100 --min-lr '1e-09' --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler inverse_sqrt \
  --max-update 20000 --warmup-updates 4000 --warmup-init-lr '1e-07' --update-freq 4\
  --adam-betas '(0.9, 0.98)' --keep-last-epochs 10 \
  --dropout 0.3 \
  --tensorboard-logdir $TRAIN_DIR/log --log-format simple\
  --save-dir $TRAIN_DIR/ckpt  \
  --eval-bleu \
  --patience 15 \
  --fp16     \   
### Step 3: scripts/average_checkpoints.py
  --inputs $TRAIN_DIR/ckpt \
  --num-epoch-checkpoints 10  \
  --output $TRAIN_DIR/ckpt/model.pt  
### Step 4: generate.py
  $DATA_DIR  
  --path $TRAIN_DIR/ckpt/model.pt \
  --source-lang $SRC_LANG \
  --target-lang $TGT_LANG \
  --beam 5 \
  --num-workers 12 \
  --batch-size 128 \
  --results-path  $TRAIN_DIR/ckpt/results2016 \
  --fp16   \
  --remove-bpe  \