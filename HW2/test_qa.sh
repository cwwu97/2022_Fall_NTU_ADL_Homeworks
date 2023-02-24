python3 -m test_qa_no_trainer --test_file ./dataset/test_relevant.json \
                             --context_file ./dataset/context.json \
                             --output_dir ./dataset \
                             --prediction_file ./prediciton.csv \
                             --do_predict \
                             --max_length 512 \
                             --pad_to_max_length \
                             --model_name_or_path ./model/qa \
                             --gradient_accumulation_steps 16 \