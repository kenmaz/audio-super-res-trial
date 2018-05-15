#!/bin/sh

python src/train.py out/ out/ data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5 data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5 -e 5
TAR_FILE=result.2018-05-16.tar
tar cvf $TAR_FILE out
aws s3 cp $TAR_FILE s3://tryswift/audio-super-resolution/.

