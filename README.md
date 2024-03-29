
# Curricular Learning

Fork of [nanoGPT](https://github.com/karpathy/nanoGPT) to experiment on with  Curricular Learning

![result](assets/result_val_loss.png)

## Install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess Wikipedia dataset)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## Prepare

```
$ python data/sorted_wikipedia_dataset/prepare.py
```

This creates a 5x Shards and `val.bin` in that data directory.

![setup](assets/experiment_sketch.png)

## Train

Now rename current shrad to be trained on to `train.bin` and remember to chang num_ters or else it will not run.
```
$ python train.py config/train_wikipedia_shards.py
```
