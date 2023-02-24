# Environment
- Create a new environment for this homework.
```shell
# Create virtual env
pythons -m venv hw1venv

# Activate virtual env
source hw1venv/bin/activate

# Install requirements
pip3 install -r requirements.in

```

# Preprocess
- Download embedding model and build vocabulary map
```shell
bash preprocess.sh
```

# Task 1: Intent Classification
- Training 
```shell
python3 -m train_intent \
            --data_dir [PATH] \
            --cache_dir [PATH] \
            --ckpt_dir [PATH] \
            --max_len [INT] \
            --hidden_size [INT] \
            --num_layers [INT] \
            --dropout [FLOAT] \
            --bidirectional [BOOLEAN] \
            --lr [FLOAT] \
            --batch_size [INT] \
            --device [STRING] \
            --num_epoch [INT] \
```
e.g.,
```shell
python3 -m train_intent --max_len 32 --hidden_size 512 --dropout 0.2 --lr 1e-3 --num_epoch 40
```

- Testing
```shell
python3 -m test_intent \
            --test_file [PATH] \
            --cache_dir [PATH] \
            --ckpt_path [PATH] \
            --pred_file [PATH] \
            --max_len [INT] \
            --hidden_size [INT] \
            --num_layers [INT] \
            --dropout [FLOAT] \
            --bidirectional [BOOLEAN] \
            --batch_size [INT] \
            --device [STRING] \
```
e.g.,
```shell
python3 -m test_intent --pred_file ./intent.pred.pt
```

# Task 2: Slot Tagging
- Training 
```shell
python3 -m train_slot \
            --data_dir [PATH] \
            --cache_dir [PATH] \
            --ckpt_dir [PATH] \
            --max_len [INT] \
            --hidden_size [INT] \
            --num_layers [INT] \
            --dropout [FLOAT] \
            --bidirectional [BOOLEAN] \
            --lr [FLOAT] \
            --batch_size [INT] \
            --device [STRING] \
            --num_epoch [INT] \
```
e.g.,
```shell
python3 -m train_intent --max_len 64 --hidden_size 512 --dropout 0.2 --lr 1e-3 --num_epoch 40
```

- Testing
```shell
python3 -m test_slot \
            --test_file [PATH] \
            --cache_dir [PATH] \
            --ckpt_path [PATH] \
            --pred_file [PATH] \
            --max_len [INT] \
            --hidden_size [INT] \
            --num_layers [INT] \
            --dropout [FLOAT] \
            --bidirectional [BOOLEAN] \
            --batch_size [INT] \
            --device [STRING] \
```
e.g.,
```shell
python3 -m test_intent --pred_file ./slot.pred.csv
```

# Run prediction script
- Assume default data path is `./data/{TASK}` 
- Intent Classification
```shell
bash ./intent_cls.sh data/intent/test.json intent.pred.csv
```
- Slot Tagging
```shell
bash ./slot_tag.sh data/slot/test.json slot.pred.csv
```