import os
import json
from metrics import get_metrics

def process_folder(input_dir, output_dir):
    """
    处理整个文件夹的JSON文件并生成评估结果
    
    Args:
        input_dir: 输入文件夹路径（包含多个json文件）
        output_dir: 输出结果目录
    """
    all_data = []
    
    # 遍历输入文件夹中的所有JSON文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_dir, filename)
            # 修改后的合并逻辑
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                # 添加文件级别的元数据（可选）
                for item in file_data:
                    item["source_file"] = filename
                all_data.extend(file_data)

    # 生成评估结果
    # 直接设置为True（根据用户确认这是ASQA数据集）
    metrics = get_metrics(all_data, save_dir=output_dir, is_asqa=True)
    return metrics

if __name__ == "__main__":
    # 直接在此处设置路径（按需修改）
    INPUT_FOLDER = "/workspace/zks_Multiagents_InstructRAG/generation_results/ASQA/ASQA_s12_t0.3_k5_p0.5_20250428_212729"  # 替换为实际输入路径
    OUTPUT_FOLDER = "/workspace/zks_Multiagents_InstructRAG/eval_results/Multiagents_InstructRAG/InstructRAG-FT/ASQA/ASQA_s12_t0.3_k5_p0.5_20250428_212729/metrics.json"  # 替换为实际输出路径
    
    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 执行评估流程
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
    print(f"评估完成！结果已保存至 {OUTPUT_FOLDER}")
