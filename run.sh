#!/bin/sh

source activate tensorflow_p27

rm -rf out
mkdir out
python src/train.py out/ out/ data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5 data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 

DATE=`date "+%Y%m%d-%H%M%S"`
TAR_FILE=result.$DATE.tar
IMG_FILE=img.$DATE.tar
S3_DIR=s3://tryswift/audio-super-resolution/$DATE

tar cvf $TAR_FILE out/events.* out/model.h5
aws s3 cp $TAR_FILE $S3_DIR/$TAR_FILE.

python src/pred.py out/model.h5 data/vctk/VCTK-Corpus/wav48/p225/p225_001.wav
python src/pred.py out/model.h5 data/vctk/VCTK-Corpus/wav48/p225/p225_002.wav

tar cvf $IMG_FILE data/vctk/VCTK-Corpus/wav48/p225/p225_001.* data/vctk/VCTK-Corpus/wav48/p225/p225_002.*
aws s3 cp $IMG_FILE $S3_DIR/$IMG_FILE
