#!/bin/bash
echo "Make sure you execute this script under the directory where you want to download the data."
# Prepare the MIND small dataset
curl -L -O https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
mkdir -p MIND_small/train
unzip MINDsmall_train.zip -d MIND_small/train
curl -L -O https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
mkdir -p MIND_small/test
unzip MINDsmall_dev.zip -d MIND_small/test

# Prepare the MIND large dataset
# curl -L -O https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip
# mkdir -p MIND_large/train
# unzip MINDlarge_train.zip -d MIND_large/train
# curl -L -O https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip
# mkdir -p MIND_large/test
# unzip MINDlarge_dev.zip -d MIND_large/test