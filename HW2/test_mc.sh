python3 -m test_mc_no_trainer --test_file ./dataset/test.json \
                             --context_file ./dataset/context.json \
                             --output_dir ./dataset \
                             --max_length 512 \
                             --pad_to_max_length \
                             --model_name_or_path ./model/mc \
                             --gradient_accumulation_steps 16 \