# Usage
1. download glove.6B.zip from https://nlp.stanford.edu/projects/glove/
2. unzip glove.6B.zip into data/ (glove.6B.300d.txt will be used)
3. run train.py
```
# Basic Pytorch environment is required
# Example
CUDA_VISIBLE_DEVICES=0 python train.py --adaptive --data data/dbp15k_zh_en
```
