# Environment
- Create a new environment for this homework.
```shell
# Create virtual env
pythons -m venv hw2venv

# Activate virtual env
source hw2venv/bin/activate

# Install requirements
pip3 install -r requirements.txt

```

# Preprocess
- Download multiple choice model and question answering model
```shell
bash download.sh
```

# Run Prediction
- 
```shell
bash run.sh [PATH TO CONTEXT FILE] [PATH TO TEST FILE] [PATH TO OUTPUT PREDICTION FILE]
```
e.g., 
- Assume defult data path is `./dataset/{FILE_NAME}.json`
```shell
bash run.sh ./dataset/context.json ./dataset/text.json ./prediction.csv
```
- You will see the prediction result in the same directory after execution.


# Model Training 
## Multiple Choice
- Training 
```shell
python3 -m run_mc_no_trainer --train_file [PATH] \
                             --validation_file [PATH] \
                             --context_file [PATH] \
                             --output_dir [PATH] \
                             --max_length [INT] \
                             --pad_to_max_length \
                             --model_name_or_path [STRING or PATH] \
                             --per_device_train_batch_size [INT] \
                             --per_device_eval_batch_size [INT] \
                             --learning_rate [FLOAT] \
                             --num_train_epochs [INT] \
                             --gradient_accumulation_steps [INT] \
                             --seed [INT]
```
e.g.,
```shell
python3 -m run_mc_no_trainer --max_length 512 \
                             --pad_to_max_length \
                             --model_name_or_path bert-base-chinese \
                             --per_device_train_batch_size 4 \
                             --per_device_eval_batch_size 4 \
                             --num_train_epochs 5 \
                             --gradient_accumulation_steps 16 \
```

- Testing
```shell
python3 -m test_mc_no_trainer --test_file [PATH] \
                              --context_file [PATH] \
                              --output_dir [PATH] \
                              --max_length [INT] \
                              --pad_to_max_length \
                              --model_name_or_path [PATH] \
```
e.g.,
```shell
python3 -m test_mc_no_trainer --output_dir ./dataset \
                              --pad_to_max_length \
                              --model_name_or_path ./model/mc \
```

## Quesiton Answering
- Training
```shell
python3 -m run_mc_no_trainer --train_file [PATH] \
                             --validation_file [PATH] \
                             --context_file [PATH] \
                             --output_dir [PATH] \
                             --max_length [INT] \
                             --pad_to_max_length \
                             --model_name_or_path [STRING or PATH] \
                             --per_device_train_batch_size [INT] \
                             --per_device_eval_batch_size [INT] \
                             --learning_rate [FLOAT] \
                             --num_train_epochs [INT] \
                             --gradient_accumulation_steps [INT] \
                             --max_answer_length [INT] \
                             --seed [INT]     
```
e.g.,
```shell
python3 -m run_mc_no_trainer --max_length 512 \
                             --pad_to_max_length \
                             --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
                             --per_device_train_batch_size 4 \
                             --per_device_eval_batch_size 4 \
                             --num_train_epochs 5 \
                             --gradient_accumulation_steps 16 \
                             --max_answer_length 128 
```

- Testing
```shell
python3 -m test_qa_no_trainer --test_file [PATH] \
                             --context_file [PATH] \
                             --prediction_file [PATH] \
                             --do_predict \
                             --max_length [INT] \
                             --pad_to_max_length \
                             --model_name_or_path [PATH] \
                             --gradient_accumulation_steps [INT] \
```
e.g.,
```shell
python3 -m test_qa_no_trainer --prediction_file ./prediciton.csv \
                              --do_predict \
                              --pad_to_max_length \
```