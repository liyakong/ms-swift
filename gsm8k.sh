SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""

ASCEND_RT_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
python -m swift.elastic_relay.runner rlhf \
    --rlhf_type grpo \
    --model /mnt/data2/lyk/traininfer/models/Qwen3.5-0.8B \
    --external_plugins examples/train/grpo/plugin/gsm8k/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --enable_thinking false \
    --truncation_strategy left \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.2 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 8192 \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --dataset '/mnt/data2/lyk/traininfer/ms-swift/ms-swift-4.5.2/ms-swift-4.5.2/grpo-datasets/gsm8k_train_messages.jsonl' \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 8192 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --report_to tensorboard swanlab \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --output_dir /output \
    --scale_rewards none
