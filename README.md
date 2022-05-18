# Noise-robust Cross-modal Interactive Learning with Text2image Mask
 Noise-robust Cross-modal Interactive Learning with Text2image Mask for Multi-modal Neural Machine Translation
## Requirements
ubuntu  
cuda==11.2  
python==3.7  
torch==1.8.1  
## dataset
txt dataWe employ the data set [Multi30K data set](http://www.statmt.org/wmt18/multimodal-task.html), then use [BPE](https://github.com/rsennrich/subword-nmt) to preprocess the raw data(dataset/data/task1/tok/). Image features are extracted through the pre-trained Resnet-101.  
##### BPE (learn_joint_bpe_and_vocab.py and apply_bpe.py)
English, German, French use BPE participle separately.   
-s 6000 \
--vocabulary-threshold 1 \
## MultimodelMixed-MMT Quickstart
Step 1: bash data-preprocess.sh  Then add the pre-trained Resnet-101 image feature to $DATA_DIR   
step 2: bash data-train.sh  
step 3: bash data-checkpoints.sh  
step 4: bash data-generate.sh  
The data-bin folder is the text data processed by bash data-preprocess.sh. Add the extracted image features here to start training the model.  
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


## Reproduce Existing Methods  
Doubly-ATT. 
[fairseq-Doubly-att.zip](https://github.com/nlp-mmt/Noise-robust-Text2image-Mask/files/8716580/fairseq-Doubly-att.zip)

Multimodal Transformer. 
[fairseq-Multimodal_Transformer.zip](https://github.com/nlp-mmt/Noise-robust-Text2image-Mask/files/8716584/fairseq-Multimodal_Transformer.zip)

Graph-based MMT. 
[fairseq-Graph-based.zip](https://github.com/nlp-mmt/Noise-robust-Text2image-Mask/files/8716585/fairseq-Graph-based.zip)


