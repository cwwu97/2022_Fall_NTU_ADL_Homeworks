# Environment
- Create a new environment for this homework.
```shell
# Create virtual env
pythons -m venv hw3venv

# Activate virtual env
source hw3venv/bin/activate

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
bash run.sh [PATH TO TEST FILE] [PATH TO OUTPUT PREDICTION FILE]
```
e.g., 
- Assume defult data path is `./data/{FILE_NAME}.json`
```shell
bash run.sh ./dataset/test.jsonl ./submission.jsonl
```
- You will see the prediction result in the same directory after execution.


# Model Training 
- Training 
```shell
python3 -m run_summarization_no_trainer --train_file [PATH] \
                                        --validation_file [PATH] \
                                        --output_dir [PATH] \
                                        --max_length [INT] \
                                        --pad_to_max_length \
                                        --num_beams [INT] \
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
python3 -m run_summarization_no_trainer --max_length 512 \
                                        --pad_to_max_length \
                                        --num_beams 4 \
                                        --model_name_or_path google/mt5-mall \
                                        --per_device_train_batch_size 8 \
                                        --per_device_eval_batch_size 8 \
                                        --num_train_epochs 15 \
                                        --gradient_accumulation_steps 1 \
```

- Testing
```shell
python3 -m test_summarization_no_trainer --test_file [PATH] \
                                         --prediction_file [PATH] \
                                         --model_name_or_path [PATH] \
                                         --num_beams [INT]
```
e.g.,
```shell
python3 -m test_summarization_no_trainer --test_file ./data/test.jsonl
                                         --prediction_file ./submission.jsonl \
                                         --model_name_or_path ./model \
                                         --num_beams 4
```