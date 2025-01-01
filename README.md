## What is this repo for
This repo is for learning basic coding about LLM, in particular, how to do SFT and alignment (RLHF and DPO). Also introduce some datasets in the code.

## How to Prepare Env
(optional) You can prepare anaconda and create an env if you prefer virtual env.
 - downlaod file: wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
 - install anaconda: bash Anaconda3-2024.06-1-Linux-x86_64.sh
 - create virtual env: conda create -n trl python=3.11
 - source ~/.bashr
 - conda activate trl

Install the dependencies:
```
pip install -r requirements.txt
```

## How to SFT 
```
python sft_chatbot.py
```

## How to align via RLHF
```
python trl_learn_batch.py
```


## How to inference
```
python eval_conv.py
```
