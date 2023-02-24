# Execution
- This project demonstrates the training and validation result of pretrained model `bert-base-uncased` on intent classfication task.
```shell
$python3 -m inten_cls --train_file [PATH] \
                       --validation_file [PATH] \
                       --max_seq_len [INT] \
                       --batch_size [INT] \
                       --num_eoich [INT] \
                       --num_label [INT]
```

e.g., 
```shell
$python3 -m inten_cls --train_file ./data/intent/train.json \
                      --validation_file ./data/intent/eval.json
```

