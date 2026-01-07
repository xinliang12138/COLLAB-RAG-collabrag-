# nohup bash eval.sh > ./eval_log/ASQA/eval_Multiagents_output3.log 2>&1 &
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 指定使用的 GPU，例如使用 GPU 0
export CUDA_VISIBLE_DEVICES=0,2,4,5
#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 数据集和模型设置
DATASET=ASQA # [PopQA, TriviaQA, NaturalQuestions, 2WikiMultiHopQA, ASQA]
MODEL=InstructRAG-ICL # [InstructRAG-FT, InstructRAG-ICL]

# # ======== 论文原参数设置========
# SEED=42            # 随机种子
# TEMPERATURE=0.7    # 温度参数
# TOP_K=50           # Top-K值
# TOP_P=0.95         # Top-P值
# # ==============================================
# ======== 生成参数设置========
SEED=12            # 随机种子
TEMPERATURE=0.3    # 温度参数
TOP_K=5            # Top-K值
TOP_P=0.5          # Top-P值
BATCH_SIZE=16       # 批处理大小
# ==============================================

# ======== 多智能体设置 ========
# 取消下面一行的注释以启用多智能体协作模式
USE_MULTIAGENT=true
# 多智能体默认温度参数(如果未指定单独温度则使用此值)
AGENT_TEMPERATURE=0.7
# 各智能体单独温度设置
DOMAIN_AGENT_TEMP=0.5  # 领域分类专家温度
EXPERTA_AGENT_TEMP=0.5  # 推理专家A温度(较高以鼓励创造性思考)
EXPERTB_AGENT_TEMP=0.5  # 推理专家B温度(适中，平衡保守与创新)

DEBATE_ROUNDS=4  # # 辩论轮数设置 (2-5轮) 默认进行3轮辩论
# ==============================================

echo "使用参数: 随机种子=$SEED, 温度=$TEMPERATURE, TOP_K=$TOP_K, TOP_P=$TOP_P"

# 输出目录：包含参数信息便于识别
EVAL_RESULTS_OUTPUT_DIR=eval_results/Multiagents_InstructRAG/${MODEL}/${DATASET}/${DATASET}_s${SEED}_t${TEMPERATURE}_k${TOP_K}_p${TOP_P}_${TIMESTAMP}

# 结果保存目录
GENERATION_RESULTS_OUTPUT_DIR=/workspace/zks_Multiagents_InstructRAG/generation_results/${DATASET}/${DATASET}_s${SEED}_t${TEMPERATURE}_k${TOP_K}_p${TOP_P}_${TIMESTAMP}

# 根据数据集自动构建模型路径
#MODEL_PATH=./model/FT-models/${DATASET}-InstructRAG-FT
#用InstructRAG-FT模式换成这个指令   --model_name_or_path "$MODEL_PATH" \
#用InstructRAG-ICL模式换成这个指令  --model_name_or_path "/workspace/zks_Multiagents_InstructRAG/model/Llama-3-8B-Instruct" \
# 运行评估脚本
python src/inference.py \
  --model_name_or_path "/workspace/zks_Multiagents_InstructRAG/model/Llama-3-8B-Instruct" \
  --dataset_name $DATASET \
  --rag_model $MODEL \
  --n_docs 5 \
  --temperature $TEMPERATURE \
  --seed $SEED \
  --top_k $TOP_K \
  --top_p $TOP_P \
  --batch_size $BATCH_SIZE \
  --output_dir $EVAL_RESULTS_OUTPUT_DIR \
  --results_dir $GENERATION_RESULTS_OUTPUT_DIR \
  --save_interval 100 \
  --load_local_model \
  --cache_dir "/workspace/zks_Multiagents_InstructRAG/model_cache" \
  $([ ! -z "$USE_MULTIAGENT" ] && echo "--use_multiagent --agent_temperature $AGENT_TEMPERATURE \
  --domain_agent_temp $DOMAIN_AGENT_TEMP \
  --expertA_agent_temp $EXPERTA_AGENT_TEMP \
  --expertB_agent_temp $EXPERTB_AGENT_TEMP \
  --debate_rounds $DEBATE_ROUNDS")

# "/workspace/zks_Multiagents_InstructRAG/model/FT-models/2WikiMultiHopQA-InstructRAG-FT"
# "./model/FT_models/NaturalQuestions-InstructRAG-FT" 
# "./model/FT_models/PopQA-InstructRAG-FT" 
# "./model/FT_models/TriviaQA-InstructRAG-FT" 
# "./model/FT_models/ASQA-InstructRAG-FT" 
# --load_local_model # Uncomment this line if you want to load a local model