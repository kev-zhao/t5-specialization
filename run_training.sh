python run_t5_mlm_flax.py \
    --model_type t5 \
    --config_name t5-small-4L-8H \
    --tokenizer_name t5-small \
    --cache_dir /mnt/home/kzhao/.cache/huggingface/ \
    --dataset_pickle_path processed_realnewslike.pkl \
    --max_seq_length 128 \
    --preprocessing_num_workers 12 \
    --output_dir t5_pretraining_jax_1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --learning_rate 0.003 \
    --weight_decay 0.001 \
    --adafactor \
    --num_train_epochs 1 \
    --warmup_steps 4000 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --logging_steps 20 \
&& \
python run_t5_mlm_flax.py \
    --model_type t5 \
    --config_name t5-small-4L-8H \
    --tokenizer_name t5-small \
    --layer_norm_type vanilla \
    --cache_dir /mnt/home/kzhao/.cache/huggingface/ \
    --dataset_pickle_path processed_realnewslike.pkl \
    --max_seq_length 128 \
    --preprocessing_num_workers 12 \
    --output_dir t5_pretraining_jax_1 \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --learning_rate 0.003 \
    --weight_decay 0.001 \
    --adafactor \
    --num_train_epochs 1 \
    --warmup_steps 4000 \
    --save_steps 2000 \
    --eval_steps 1000 \
    --logging_steps 20
