DATASET=PopQA
PER_DEVICE_BATCH_SIZE=1
NUM_DEVICE=5
TOTAL_BATCH_SIZE=20
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_DEVICE/$PER_DEVICE_BATCH_SIZE))

CUDA_VISIBLE_DEVICES="0,1,4,5,6" torchrun --nproc_per_node=$NUM_DEVICE src/finetune.py \
  --model_name_or_path /workspace/zks_InstructRAG/model/Llama-3-8B-Instruct \
  --dataset_name $DATASET \
  --output_dir /workspace/zks_InstructRAG/saved_checkpoints/InstructRAG-FT/${DATASET} \
  --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --num_train_epochs 2 \
  --n_docs 5 \
  --learning_rate 2.5e-5 \
  --lr_scheduler_type "cosine" \
  --bf16 False \
  --tf32 False \
  --fp16 True \
  --logging_steps 1 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --seed 42 \
  --model_max_length 4096 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_config '{"activation_checkpointing": true, "transformer_layer_cls_to_wrap": "LlamaDecoderLayer"}'

# # 注意：移除了多卡相关参数，使用绝对路径更可靠
# CUDA_VISIBLE_DEVICES="0" python src/finetune.py \
#   --model_name_or_path /root/autodl-tmp/model/Llama-3-8B-Instruct \
#   --dataset_name PopQA \
#   --output_dir ./checkpoints \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 8 \
#   --num_train_epochs 2 \
#   --n_docs 2 \
#   --learning_rate 1e-5 \
#   --bf16 True \
#   --gradient_checkpointing \
#   --load_in_4bit \
#   --bnb_4bit_compute_dtype bfloat16 \
#   --model_max_length 1024 \
#   --logging_steps 10


# DATASET=PopQA
# PER_DEVICE_BATCH_SIZE=1
# NUM_DEVICE=1
# TOTAL_BATCH_SIZE=8
# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_DEVICE/$PER_DEVICE_BATCH_SIZE))

# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=$NUM_DEVICE src/finetune.py \
#   --model_name_or_path "/root/autodl-tmp/model/Llama-3-8B-Instruct" \
#   --dataset_name $DATASET \
#   --output_dir saved_checkpoints/InstructRAG-FT/${DATASET} \
#   --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
#   --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
#   --num_train_epochs 2 \
#   --n_docs 3 \
#   --learning_rate 1e-5 \
#   --lr_scheduler_type "cosine" \
#   --bf16 True \
#   --tf32 True \
#   --logging_steps 1 \
#   --weight_decay 0.0 \
#   --warmup_ratio 0.03 \
#   --seed 42 \
#   --model_max_length 2048 \
#   --ddp_timeout 1800 \
#   --fsdp "full_shard auto_wrap" \
#   --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"