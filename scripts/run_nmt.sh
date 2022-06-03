python cli/run_summarization_flax.py \
	--output_dir nmt_test \
	--model_type t5 \
    --config_name configs/t5-small-4L-8H \
    --tokenizer_name t5-small \
    --cache_dir /home/kzhao/.cache/huggingface/ \
	--dataset_name="wmt19" \
	--dataset_config_name="de-en" \
	--do_train --do_eval --predict_with_generate \
	--num_beams 3 \
	--eval_steps 2000 \
	--num_train_epochs 1 \
	--learning_rate 5e-5 --warmup_steps 5000 \
	--per_device_train_batch_size 64 \
	--per_device_eval_batch_size 64 \
	--overwrite_output_dir \
	--max_source_length 512 --max_target_length 64 \