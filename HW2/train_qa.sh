python3 -m run_mc_no_trainer --train_file ./dataset/train.json
                             --validation_file ./dataset/validation.json
                             --context_file ./dataset/context.json \
                             --output_dir ./model/qa \
                             --max_length 512 \
                             --pad_to_max_length \
                             --model_name_or_path bert-base-chinese \
                             --per_device_train_batch_size 4 \
                             --per_device_eval_batch_size 4 \
                             --learning_rate 5e-5 \
                             --num_train_epochs 5 \ 
                             --gradient_accumulation_steps 16 \
                             --max_answer_length 128 \
                             --seed 42                              