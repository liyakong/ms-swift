PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=1,2 \
NPROC_PER_NODE=2 \
megatron sft \
    --model /data/AI/models/Qwen3-4B \
    --save_safetensors true \
    --dataset /data/AI/lyk/ms-swift/ms-swift-4.5.2/train.csv \
    --use_hf true \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --packing True \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 5 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --output_dir megatron_output/Qwen3-4B \
    --eval_steps 500 \
    --save_steps 500 \
    --max_length 8192 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --cross_entropy_fusion_impl native \
    --attention_backend flash
