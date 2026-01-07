# import os
# import sys
# import argparse
# import data_utils
# import common_utils
# from metrics import get_metrics
# from vllm import LLM, SamplingParams



# def generate_rationale(args):
#     data_path = f'dataset/{args.dataset_name}/train.json'
#     print(f"Loading training set from: {data_path}")
#     train_data = common_utils.jload(data_path)[:args.max_instances]

#     llm = LLM(model=args.model_name_or_path, download_dir=args.cache_dir, max_model_len=args.max_tokens)

#     tokenizer = llm.get_tokenizer()
#     prompt_dict = common_utils.jload(args.prompt_dict_path)

#     prompts = data_utils.format_prompt_with_data_list(
#         data_list=train_data,
#         dataset_name=args.dataset_name,
#         prompt_dict=prompt_dict,
#         tokenizer=tokenizer,
#         n_docs=args.n_docs,
#         do_rationale_generation=True,
#     )

#     sampling_params = SamplingParams(temperature=args.temperature, 
#                                     max_tokens=args.max_tokens, 
#                                     seed=args.seed,
#                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
#     outputs = llm.generate(prompts, sampling_params)

#     output_file = os.path.join(args.output_dir, "with_rationale/train.json")

#     save_outputs(outputs, train_data, output_file, args.n_docs)

# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# parser.add_argument('--load_local_model', type=str, default=None, help="Path to SINGLE local fine-tuned model")
# parser.add_argument('--load_local_models', nargs='+', default=[], help="List of paths to MULTIPLE local models")

# def eval_model(args):
#     data_path = f'dataset/{args.dataset_name}/test.json'
#     print(f"Loading eval set from: {data_path}")
#     test_data = common_utils.jload(data_path)[:args.max_instances]

#     print(f'Loading model {args.rag_model}...')
#     if args.rag_model == 'InstructRAG-FT':
#         demos = []
#         # 优先级：多模型 > 单模型 > 云端默认模型
#         if args.load_local_models:  # 多模型模式
#             llms = [LLM(model=path, max_model_len=args.max_tokens) 
#                 for path in args.load_local_models]
#             print(f"Loaded {len(llms)} local models")
#         elif args.load_local_model:  # 单模型模式
#             llm = LLM(model=args.load_local_model, max_model_len=args.max_tokens)
#             print(f"Loaded local model from {args.load_local_model}")
#         else:  # 默认云端模型
#             llm = LLM(model=f'meng-lab/{args.dataset_name}-InstructRAG-FT', download_dir=args.cache_dir, max_model_len=args.max_tokens)
            
#     elif args.rag_model == 'InstructRAG-ICL':
#          demos = common_utils.jload(f'dataset/{args.dataset_name}/demos.json')
#          llm = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct', download_dir=args.cache_dir, max_model_len=args.max_tokens)

#     # if args.rag_model == 'InstructRAG-FT':
#     #     demos = []
#     #     if args.load_local_model:
#     #         llm = LLM(model=f'saved_checkpoints/InstructRAG-FT/{args.dataset_name}',  max_model_len=args.max_tokens)
#     #     else:
#     #         llm = LLM(model=f'meng-lab/{args.dataset_name}-InstructRAG-FT', download_dir=args.cache_dir, max_model_len=args.max_tokens)
#     # elif args.rag_model == 'InstructRAG-ICL':
#     #     demos = common_utils.jload(f'dataset/{args.dataset_name}/demos.json')
#     #     llm = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct', download_dir=args.cache_dir, max_model_len=args.max_tokens)
# #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#     tokenizer = llm.get_tokenizer()
#     prompt_dict = common_utils.jload(args.prompt_dict_path)
 
#     prompts = data_utils.format_prompt_with_data_list(
#         data_list=test_data,
#         dataset_name=args.dataset_name,
#         prompt_dict=prompt_dict,
#         tokenizer=tokenizer,
#         n_docs=args.n_docs,
#         demos=demos,
#     )
    
#     sampling_params = SamplingParams(temperature=args.temperature, 
#                                     max_tokens=args.max_tokens, 
#                                     seed=args.seed,
#                                     stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
#     outputs = llm.generate(prompts, sampling_params)

#     output_file = os.path.join(args.output_dir, "result.json")

#     eval_results = save_outputs(outputs, test_data, output_file, args.n_docs)
#     get_metrics(eval_results, args.output_dir, is_asqa=args.dataset_name == 'ASQA')

# def save_outputs(outputs, test_data, output_file, n_docs):
#     # Save the outputs as a JSON file.
#     output_data = []
#     for i, output in enumerate(outputs):
#         prompt = output.prompt
#         generated_text = output.outputs[0].text
#         sample = test_data[i]
#         output_data.append({
#             "question": sample["question"],
#             "answers": sample["answers"],
#             "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
#             "rationale": generated_text,
#             "prompt": prompt,
#             "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
#             })
        
#     common_utils.jdump(output_data, output_file)
#     print(f"Outputs saved to {output_file}")

#     return output_data

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
#     parser.add_argument('--rag_model', type=str, choices=['InstructRAG-FT', 'InstructRAG-ICL'], default='InstructRAG-FT', help='InstructRAG model: InstructRAG-FT or InstructRAG-ICL')
#     parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='name of the model in Hugging Face model hub or path to the model')
#     parser.add_argument('--load_local_model', action='store_true', help='Load local model')
#     parser.add_argument('--do_rationale_generation', action='store_true', help='Generate rationales on training data')
#     parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
#     parser.add_argument('--output_dir', type=str, help='Path to the output file')
#     parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cached models')
#     parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
#     parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
#     parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens')
#     parser.add_argument('--seed', type=int, default=42, help='Random seed')
#     parser.add_argument('--max_instances', type=int, default=sys.maxsize)

#     args = parser.parse_args()

#     if args.do_rationale_generation:
#         generate_rationale(args)
#     else:
#         eval_model(args)


import os
import sys
import data_utils
import common_utils
from metrics import get_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random
import time
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # 添加 dtype 参数
    parser.add_argument('--dtype', type=str, default='float16', 
                        choices=['float16', 'float32', 'bfloat16'],
                        help='Data type for model weights')
    
    # 添加 batch_size 参数
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
#=====================================================================================================     
    # 添加多智能体相关参数
    parser.add_argument('--use_multiagent', action='store_true', help='Enable multi-agent collaboration')
    parser.add_argument('--agent_temperature', type=float, default=None, help='Temperature for agent generation (default)')
    parser.add_argument('--domain_agent_temp', type=float, default=None, help='Temperature for domain classifier agent')
    parser.add_argument('--expertA_agent_temp', type=float, default=None, help='Temperature for reasoning expert A')
    parser.add_argument('--expertB_agent_temp', type=float, default=None, help='Temperature for reasoning expert B')
    parser.add_argument('--debate_rounds', type=int, default=1, choices=range(2, 6), help='Number of debate rounds (2-5)')
    # 添加中间结果保存参数
    parser.add_argument('--results_dir', type=str, default=None, help='Directory to save processing results')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save intermediate results (n samples)')
#=====================================================================================================     
    # 原有参数保持不变
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--rag_model', type=str, choices=['InstructRAG-FT', 'InstructRAG-ICL'], default='InstructRAG-FT', help='InstructRAG model type')
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='name of the model in Hugging Face model hub or path to the model')
    parser.add_argument('--load_local_model', action='store_true', help='Load local model')
    parser.add_argument('--do_rationale_generation', action='store_true', help='Generate rationales on training data')
    parser.add_argument('--n_docs', type=int, default=5, help='Number of retrieved documents')
    parser.add_argument('--output_dir', type=str, help='Path to the output file')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to cached models')
    parser.add_argument('--prompt_dict_path', type=str, default="src/rag.json")
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for sampling')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_instances', type=int, default=sys.maxsize)
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling parameter')
    
    return parser.parse_args()

class TransformersModel:
    def __init__(self, model_path, cache_dir=None, max_length=4096, batch_size=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            padding_side="left",
            truncation_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            rope_scaling=None,  #################################删除会触发rope_scaling警告
            pad_token_id=self.tokenizer.pad_token_id
        )
        self.max_length = max_length

    def generate(self, prompts, sampling_params, batch_size=None):
        outputs = []
        
        # 使用传入的batch_size或默认值
        if batch_size is None:
            batch_size = self.batch_size
        
        total_prompts = len(prompts)
        
        # 批量处理提示，不显示额外进度条
        for i in range(0, total_prompts, batch_size):
            # 计算实际批量大小(可能小于batch_size，如果是最后一批)
            actual_batch_size = min(batch_size, total_prompts - i)
            batch_prompts = prompts[i:i+actual_batch_size]
            
            inputs = self.tokenizer(
                batch_prompts, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=sampling_params.max_tokens,
                    do_sample=sampling_params.temperature > 0,
                    temperature=sampling_params.temperature,
                    top_k=sampling_params.top_k,
                    top_p=sampling_params.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=sampling_params.stop_token_ids[0] if sampling_params.stop_token_ids else None
                )
            
            # 处理批量输出
            for j, output_id in enumerate(output_ids):
                generated_text = self.tokenizer.decode(output_id, skip_special_tokens=True)
                prompt = batch_prompts[j]
                output = type('GenerationOutput', (), {
                    'prompt': prompt,
                    'outputs': [type('Output', (), {'text': generated_text})]
                })
                outputs.append(output)
        
        return outputs

    def get_tokenizer(self):
        return self.tokenizer

class SamplingParams:
    def __init__(self, temperature=0, max_tokens=100, seed=42, top_k=50, top_p=0.95, stop_token_ids=None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.top_k = top_k
        self.top_p = top_p
        self.stop_token_ids = stop_token_ids if stop_token_ids else []
#=====================================================================================================
# 添加多智能体处理函数
def run_multiagent_analysis(model, tokenizer, prompt_dict, example, docs_text, 
                            domain_sampling_params, expertA_sampling_params, expertB_sampling_params,
                            max_debate_rounds=2):
    """执行多智能体分析流程 - 专家辩论式解决方案，适用于有确定答案的各类问题"""
    # 获取角色指令
    role_instructions = prompt_dict['role_instructions']
    
    # 1. 领域分类专家 - 使用domain_sampling_params
    domain_classifier_prompt = data_utils.build_role_prompt({
        "system_prompt": "You are a professional domain classification expert. " + role_instructions['domain_classifier']
    }, query=example['question'], context=docs_text)
    
    domain_outputs = model.generate([domain_classifier_prompt], domain_sampling_params)
    domain_analysis = domain_outputs[0].outputs[0].text.strip()
    
    # 2. 直接进行推理专家A和B的分析 - 并行准备提示
    expertA_prompt = data_utils.build_role_prompt({
        "system_prompt": "You are Reasoning Expert A. Your task is to carefully analyze the question and context, and provide a clear answer. "
                         "You must explain your reasoning process in detail and clearly state your answer in the conclusion. "
                         "Please provide your answer in the format 'My answer is: XXX'. " 
                         + role_instructions['reasoning_expert']
    }, query=example['question'], context=docs_text, prev_analysis=domain_analysis)
    
    expertB_prompt = data_utils.build_role_prompt({
        "system_prompt": "You are Reasoning Expert B. Your task is to think independently, carefully analyze the question and context, and provide a clear answer. "
                         "You must explain your reasoning process in detail and clearly state your answer in the conclusion. "
                         "Please provide your answer in the format 'My answer is: XXX'. "
                         + role_instructions['reasoning_expert']
    }, query=example['question'], context=docs_text, prev_analysis=domain_analysis)
    
    # 获取A和B的回答 - 使用各自的温度参数分别生成
    expertA_output = model.generate([expertA_prompt], expertA_sampling_params)[0]
    expertB_output = model.generate([expertB_prompt], expertB_sampling_params)[0]
    
    expertA_reasoning = expertA_output.outputs[0].text.strip()
    expertB_reasoning = expertB_output.outputs[0].text.strip()
    
    # 尝试从专家A的回答中提取答案
    expertA_answer = extract_answer(expertA_reasoning)
    if not expertA_answer:
        expertA_answer = "No clear answer provided"
    
    # 尝试从专家B的回答中提取答案
    expertB_answer = extract_answer(expertB_reasoning)
    if not expertB_answer:
        expertB_answer = "No clear answer provided"
    
    # 记录第一轮辩论
    debate_history = [{
        "round": 1,
        "expertA": expertA_reasoning,
        "expertA_answer": expertA_answer,
        "expertB": expertB_reasoning,
        "expertB_answer": expertB_answer
    }]
    
    # 检查是否达成共识
    is_consensus = False
    final_answer = None
    current_round = 1
    
    if are_answers_equivalent(expertA_answer, expertB_answer) and expertA_answer != "No clear answer provided":
        is_consensus = True
        final_answer = expertA_answer
    else:
        # 如果第一轮没有达成共识，且需要多轮辩论，继续辩论
        while not is_consensus and current_round < max_debate_rounds:
            current_round += 1
            
            # Expert A reconsiders after seeing Expert B's opinion
            debate_summary = f"Your previous analysis and answer was: {expertA_reasoning}\n\nThe other expert's analysis and answer was: {expertB_reasoning}"
            expertA_prompt = data_utils.build_role_prompt({
                "system_prompt": "You are Reasoning Expert A. Expert B has a different perspective on this question. "
                               "Please carefully consider their reasoning, then decide whether to modify your answer. "
                               "Explain your reasoning process in detail and clearly state your answer in the conclusion. "
                               "Please provide your answer in the format 'My answer is: XXX'. "
                               "If you change your mind, explicitly state why. If you maintain your original answer, explain why your analysis is more reasonable."
            }, query=example['question'], context=docs_text, prev_analysis=domain_analysis, analyses=debate_summary)
            
            # Expert B reconsiders after seeing Expert A's opinion
            debate_summary = f"Your previous analysis and answer was: {expertB_reasoning}\n\nThe other expert's analysis and answer was: {expertA_reasoning}"
            expertB_prompt = data_utils.build_role_prompt({
                "system_prompt": "You are Reasoning Expert B. Expert A has a different perspective on this question. "
                               "Please carefully consider their reasoning, then decide whether to modify your answer. "
                               "Explain your reasoning process in detail and clearly state your answer in the conclusion. "
                               "Please provide your answer in the format 'My answer is: XXX'. "
                               "If you change your mind, explicitly state why. If you maintain your original answer, explain why your analysis is more reasonable."
            }, query=example['question'], context=docs_text, prev_analysis=domain_analysis, analyses=debate_summary)
            
            # 使用各自的温度参数分别生成新回答
            expertA_output = model.generate([expertA_prompt], expertA_sampling_params)[0]
            expertB_output = model.generate([expertB_prompt], expertB_sampling_params)[0]
            
            # 更新专家推理和答案
            expertA_reasoning = expertA_output.outputs[0].text.strip()
            expertB_reasoning = expertB_output.outputs[0].text.strip()
            
            # 提取新答案
            previous_A_answer = expertA_answer
            expertA_answer = extract_answer(expertA_reasoning)
            if not expertA_answer:
                expertA_answer = previous_A_answer
                
            previous_B_answer = expertB_answer
            expertB_answer = extract_answer(expertB_reasoning)
            if not expertB_answer:
                expertB_answer = previous_B_answer
            
            # 更新辩论历史
            debate_history.append({
                "round": current_round,
                "expertA": expertA_reasoning,
                "expertA_answer": expertA_answer,
                "expertB": expertB_reasoning,
                "expertB_answer": expertB_answer
            })
            
            # Check if consensus is reached
            if are_answers_equivalent(expertA_answer, expertB_answer) and expertA_answer != "No clear answer provided":
                is_consensus = True
                final_answer = expertA_answer
        
        # 辩论结束，如果没有达成共识，使用专家A的答案
        if not is_consensus:
            final_answer = expertA_answer
    
    # 生成辩论摘要
    debate_summary = format_debate_summary(debate_history, final_answer)
    
    # 整合分析结果
    example['domain_analysis'] = domain_analysis
    example['debate_summary'] = debate_summary
    example['final_answer'] = final_answer
    
    return example

def extract_answer(text):
    """从专家回答中提取答案"""
    import re
    
    # 尝试匹配"我的答案是：XXX"格式
    patterns = [
        # English explicit answer formats
        r'my answer is[：:]\s*(.+?)[.!?\n]',
        r'the answer is[：:]\s*(.+?)[.!?\n]',
        r'answer[：:]\s*(.+?)[.!?\n]',
        r'i answer[：:]\s*(.+?)[.!?\n]',
        r'my response is[：:]\s*(.+?)[.!?\n]',
        r'the correct answer is[：:]\s*(.+?)[.!?\n]',
        r'the final answer is[：:]\s*(.+?)[.!?\n]',
        r'my conclusion is[：:]\s*(.+?)[.!?\n]',
        r'i conclude that[：:]\s*(.+?)[.!?\n]',
        r'i believe the answer is[：:]*\s*(.+?)[.!?\n]',
        
        # Formats without punctuation ending
        r'my answer is[：:]\s*(.+?)$',
        r'the answer is[：:]\s*(.+?)$',
        r'the conclusion is[：:]\s*(.+?)$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # 如果提取的答案超过70个字符，可能是误匹配了一段长文本，尝试只保留首句
            if len(answer) > 70:
                # Try to truncate at first period
                first_sentence = re.split(r'[.!?\n]', answer)[0]
                if len(first_sentence) > 5:  # Ensure it's not too short
                    return first_sentence.strip()
            return answer
    
    # If no explicit format found, try to extract from conclusion section
    conclusion_patterns = [
        r'in conclusion,?\s*(.+?)[.!?\n]',
        r'to summarize,?\s*(.+?)[.!?\n]',
        r'in summary,?\s*(.+?)[.!?\n]',
        r'therefore,?\s*(.+?)[.!?\n]',
        r'thus,?\s*(.+?)[.!?\n]',
        r'finally,?\s*(.+?)[.!?\n]',
        r'in essence,?\s*(.+?)[.!?\n]',
        r'overall,?\s*(.+?)[.!?\n]',
        r'based on the above,?\s*(.+?)[.!?\n]',
    ]
    
    for pattern in conclusion_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # 太长的答案可能是误匹配，尝试进一步处理
            if len(answer) > 70:
                # Try to extract key phrases from long conclusions
                key_content = re.search(r'the answer is[：:]\s*(.+?)[.!?\n]', answer, re.IGNORECASE)
                if key_content:
                    return key_content.group(1).strip()
                # Take first sentence
                first_sentence = re.split(r'[.!?\n]', answer)[0]
                if len(first_sentence) > 5:
                    return first_sentence.strip()
            return answer
    
    # Try to extract directly from the last few lines as a potential answer
    # Usually applicable when experts give concise answers in the summary section
    last_lines = text.strip().split('\n')[-3:]  # Take last 3 lines
    
    for line in last_lines:
        line = line.strip()
        if 5 < len(line) < 50 and re.search(r'[.!?]$', line):  # Line of appropriate length
            return line.rstrip('.!?')
    
    # 最后的备选方案：从最后一段中提取第一个短句
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if paragraphs:
        last_paragraph = paragraphs[-1]
        sentences = re.split(r'[.!?]', last_paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if 5 < len(sentence) < 50:  # Select sentence of appropriate length
                return sentence
    
    return None

def are_answers_equivalent(answer1, answer2):
    """检查两个答案是否等价（考虑同义表达和轻微差异）"""
    if answer1 is None or answer2 is None:
        return False
    
    # 如果完全相同
    if answer1.lower() == answer2.lower():
        return True
    
    # 修剪标点符号和多余空格进行比较
    import re
    clean1 = re.sub(r'[.,;:!?，。；：！？\s]+', ' ', answer1).strip().lower()
    clean2 = re.sub(r'[.,;:!?，。；：！？\s]+', ' ', answer2).strip().lower()
    
    if clean1 == clean2:
        return True
    
    # 检查一个是否是另一个的子字符串
    if clean1 in clean2 or clean2 in clean1:
        # 如果长度差距不大，认为是等价的
        min_len = min(len(clean1), len(clean2))
        max_len = max(len(clean1), len(clean2))
        if min_len > 5 and min_len / max_len > 0.7:  # 长度至少为5且比率>0.7
            return True
    
    # TODO: 可以添加更复杂的语义相似度比较，如使用嵌入向量计算相似度
    
    return False

def format_debate_summary(debate_history, final_answer):
    """Format the debate process summary"""
    summary = "Debate Process Summary:\n\n"
    
    # 跟踪专家答案变化情况
    initial_expertA_answer = debate_history[0]["expertA_answer"]
    initial_expertB_answer = debate_history[0]["expertB_answer"]
    final_expertA_answer = debate_history[-1]["expertA_answer"]
    final_expertB_answer = debate_history[-1]["expertB_answer"]
    
    # 判断是否达成共识
    consensus = are_answers_equivalent(final_expertA_answer, final_expertB_answer)
    
    # 判断专家是否改变了意见
    expertA_changed = not are_answers_equivalent(initial_expertA_answer, final_expertA_answer)
    expertB_changed = not are_answers_equivalent(initial_expertB_answer, final_expertB_answer)
    
    for debate in debate_history:
        round_num = debate["round"]
        summary += f"Round {round_num} Debate:\n"
        # Truncate expert reasoning to keep summary concise
        expertA_brief = debate['expertA'][:150] + "..." if len(debate['expertA']) > 150 else debate['expertA']
        expertB_brief = debate['expertB'][:150] + "..." if len(debate['expertB']) > 150 else debate['expertB']
        
        summary += f"Expert A: '{debate['expertA_answer']}' - {expertA_brief}\n"
        summary += f"Expert B: '{debate['expertB_answer']}' - {expertB_brief}\n\n"
    
    # Add debate status summary
    if consensus:
        summary += f"The experts reached consensus, agreeing on the answer: {final_answer}\n"
    else:
        summary += f"The experts did not reach consensus, using Expert A's answer: {final_answer}\n"
    
    # Add opinion change information
    if expertA_changed:
        summary += f"Expert A changed their initial view from '{initial_expertA_answer}' to '{final_expertA_answer}'\n"
    if expertB_changed:
        summary += f"Expert B changed their initial view from '{initial_expertB_answer}' to '{final_expertB_answer}'\n"
    
    summary += f"\nFinal conclusion: {final_answer}"
    return summary
#=====================================================================================================
def eval_model(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    data_path = f'dataset/{args.dataset_name}/test.json'
    print(f"Loading eval set from: {data_path}")
    test_data = common_utils.jload(data_path)[:args.max_instances]

    print(f"Loaded {len(test_data)} test samples")
    if len(test_data) > 0:
        print("Sample data structure:", test_data[0].keys())
    else:
        print("WARNING: test_data is empty!")

    print(f'Loading model {args.rag_model}...')
    if args.rag_model == 'InstructRAG-FT':
        demos = []
        # 新增逻辑：优先使用命令行指定的模型路径
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            print(f"Using command line specified model path: {args.model_name_or_path}")
            model_path = args.model_name_or_path
        # 原有逻辑
        elif args.load_local_model:
            #model_path = f'saved_checkpoints/InstructRAG-FT/{args.dataset_name}'
            #model_path = f'/workspace/zks_Multiagents_InstructRAG/model/FT-models/{args.dataset_name}-InstructRAG-FT'
            model_path = f'./model/FT-models/{args.dataset_name}-InstructRAG-FT'
            # 检查路径是否存在
            if not os.path.exists(model_path):
                print(f"Warning: local model path {model_path} does not exist, trying to load from Hugging Face...")
                model_path = f'meng-lab/{args.dataset_name}-InstructRAG-FT'
        else:
            model_path = f'meng-lab/{args.dataset_name}-InstructRAG-FT'
    elif args.rag_model == 'InstructRAG-ICL':
        demos = common_utils.jload(f'dataset/{args.dataset_name}/demos.json')
        
        # 新增逻辑：优先使用命令行指定的模型路径
        if args.model_name_or_path and os.path.exists(args.model_name_or_path):
            print(f"Using command line specified model path: {args.model_name_or_path}")
            model_path = args.model_name_or_path
        # 原有逻辑
        else:
            model_path = args.model_name_or_path  #这句代码如果取消注释，在eval.sh中要加上--model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct"
            
            # if args.load_local_model:
            #     # 使用固定的本地模型路径，不管数据集是哪个
            #     model_path = "./model/Llama-3-8B-Instruct"
            #     # 检查路径是否存在
            #     if not os.path.exists(model_path):
            #         print(f"警告: 本地模型路径 {model_path} 不存在，尝试从其他位置加载...")
            # else:
            #     model_path = args.model_name_or_path
    # 使用新的 TransformersModel 类
    llm = TransformersModel(model_path, args.cache_dir, args.max_tokens)
    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)
#=====================================================================================================    
    # 创建用于保存结果的列表
    final_results = []
    
    # 确定结果保存目录
    results_dir = args.results_dir if args.results_dir else args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # 流式处理每个样本
    for i, sample in enumerate(tqdm(test_data, desc="Processing samples")):
        # 创建当前样本的副本以便处理
        current_sample = sample.copy()
        
        # 如果启用了多智能体，先执行多智能体分析
        if args.use_multiagent:
            # 设置各智能体的温度参数
            domain_temp = args.domain_agent_temp if args.domain_agent_temp is not None else args.agent_temperature
            expertA_temp = args.expertA_agent_temp if args.expertA_agent_temp is not None else args.agent_temperature
            expertB_temp = args.expertB_agent_temp if args.expertB_agent_temp is not None else args.agent_temperature
            
            # 为当前样本准备上下文
            if len(current_sample["ctxs"]) > 0 and current_sample["ctxs"][0]["score"] > current_sample["ctxs"][1]["score"]:
                ctxs_list = current_sample["ctxs"][:args.n_docs][::-1]
            else:
                ctxs_list = current_sample["ctxs"][:args.n_docs]
            
            docs_text = "\n\n".join([f"Document {idx+1} (Title: {ctx['title']}): {ctx['text']}" 
                                   for idx, ctx in enumerate(ctxs_list)])
            
            # 创建sampling参数
            domain_sampling_params = SamplingParams(
                temperature=domain_temp,
                max_tokens=1024,
                seed=args.seed,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_token_ids=[tokenizer.eos_token_id]
            )
            
            expertA_sampling_params = SamplingParams(
                temperature=expertA_temp,
                max_tokens=1024,
                seed=args.seed,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_token_ids=[tokenizer.eos_token_id]
            )
            
            expertB_sampling_params = SamplingParams(
                temperature=expertB_temp,
                max_tokens=1024,
                seed=args.seed,
                top_k=args.top_k,
                top_p=args.top_p,
                stop_token_ids=[tokenizer.eos_token_id]
            )
            
            # 执行多智能体分析
            current_sample = run_multiagent_analysis(
                llm, 
                tokenizer, 
                prompt_dict, 
                current_sample, 
                docs_text, 
                domain_sampling_params,  # 领域专家参数
                expertA_sampling_params,  # 专家A参数
                expertB_sampling_params,  # 专家B参数
                max_debate_rounds=args.debate_rounds  # 传递辩论轮数
            )
#=====================================================================================================        
        # 为当前样本格式化提示词
        prompt = data_utils.format_prompt(
            dataset_name=args.dataset_name,
            example=current_sample,
            n_docs=args.n_docs,
            prompt_dict=prompt_dict,
            tokenizer=tokenizer,
            demos=demos,
            do_rationale_generation=False
        )
        
        # 为当前样本生成回复
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            top_k=args.top_k,
            top_p=args.top_p,
            stop_token_ids=[tokenizer.eos_token_id]
        )
        
        output = llm.generate([prompt], sampling_params)[0]
        
        # 构建结果项并添加到final_results
        result_item = {
            "question": current_sample["question"],
            "answers": current_sample["answers"],
            "qa_pairs": current_sample["qa_pairs"] if "qa_pairs" in current_sample else None,
            "rationale": output.outputs[0].text,
            "prompt": output.prompt,
            "ctxs": current_sample["ctxs"][:args.n_docs][::-1] if (current_sample["ctxs"][0]['score'] > current_sample["ctxs"][1]['score']) else current_sample["ctxs"][:args.n_docs],
        }
        
        # 添加多智能体结果（如果有）
        if args.use_multiagent:
            result_item["domain_analysis"] = current_sample.get("domain_analysis", "")
            result_item["debate_summary"] = current_sample.get("debate_summary", "")
            result_item["final_answer"] = current_sample.get("final_answer", "")
        
        final_results.append(result_item)
        
        # 每处理指定数量的样本保存一次中间结果
        if (i+1) % args.save_interval == 0:
            temp_output_file = os.path.join(results_dir, f"results_{i+1}.json")
            common_utils.jdump(final_results, temp_output_file)

    
    # 保存最终结果
    final_output_file = os.path.join(results_dir, "results_final.json")
    common_utils.jdump(final_results, final_output_file)
    print(f"All results saved to {final_output_file}")
    
    # 同时保存一份到输出目录中供评估使用
    output_file = os.path.join(args.output_dir, "result.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    common_utils.jdump(final_results, output_file)
    
    # 计算评估指标
    get_metrics(final_results, args.output_dir, is_asqa=args.dataset_name == 'ASQA',
               generation_params={
                   "seed": args.seed,
                   "temperature": args.temperature,
                   "top_k": args.top_k,
                   "top_p": args.top_p,
                   "batch_size": args.batch_size,
                   "use_multiagent": args.use_multiagent if hasattr(args, 'use_multiagent') else False,
                   # 添加多智能体参数
                   "multiagent_params": {
                       "agent_temperature": args.agent_temperature,
                       "domain_agent_temp": args.domain_agent_temp if args.domain_agent_temp is not None else args.agent_temperature,
                       "expertA_agent_temp": args.expertA_agent_temp if args.expertA_agent_temp is not None else args.agent_temperature,
                       "expertB_agent_temp": args.expertB_agent_temp if args.expertB_agent_temp is not None else args.agent_temperature,
                       "debate_rounds": args.debate_rounds
                   } if args.use_multiagent else None
               })

def generate_rationale(args):
    data_path = f'dataset/{args.dataset_name}/train.json'
    print(f"Loading training set from: {data_path}")
    train_data = common_utils.jload(data_path)[:args.max_instances]

    llm = TransformersModel(args.model_name_or_path, args.cache_dir, args.max_tokens)
    tokenizer = llm.get_tokenizer()
    prompt_dict = common_utils.jload(args.prompt_dict_path)

    prompts = data_utils.format_prompt_with_data_list(
        data_list=train_data,
        dataset_name=args.dataset_name,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        do_rationale_generation=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    outputs = llm.generate(prompts, sampling_params)
    output_file = os.path.join(args.output_dir, "with_rationale/train.json")
    save_outputs(outputs, train_data, output_file, args.n_docs)

def save_outputs(outputs, test_data, output_file, n_docs):
    output_data = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        sample = test_data[i]
        output_data.append({
            "question": sample["question"],
            "answers": sample["answers"],
            "qa_pairs": sample["qa_pairs"] if "qa_pairs" in sample else None,
            "rationale": generated_text,
            "prompt": prompt,
            "ctxs": sample["ctxs"][:n_docs][::-1] if (sample["ctxs"][0]['score'] > sample["ctxs"][1]['score']) else sample["ctxs"][:n_docs],
            })
        
    common_utils.jdump(output_data, output_file)
    print(f"Outputs saved to {output_file}")
    return output_data

if __name__ == "__main__":
    args = parse_args()
    if args.do_rationale_generation:
        generate_rationale(args)
    else:
        eval_model(args)

# 替换 vLLM 的 LLM 类
def load_model(model_path, dtype="float16"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if dtype == "float16" else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# # 在生成最终答案处添加
# result_entry = {
#     "final_answer": generated_text.replace("XXX", "").strip()  # 替换实际生成内容
# }