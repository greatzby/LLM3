"""
通用工具函数
"""
import os
import pickle
import numpy as np
import torch

def load_meta(data_path):
    """加载meta信息"""
    with open(os.path.join(data_path, 'meta.pkl'), 'rb') as f:
        return pickle.load(f)

def convert_to_serializable(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def load_model(checkpoint_path, device='cuda:0'):
    """加载模型 - 通用版本"""
    from model import GPTConfig, GPT
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

def decode_tokens(token_ids, itos):
    """解码token序列为字符串"""
    decoded = []
    for tid in token_ids:
        if tid == 1:  # 换行符
            decoded.append('\n')
            break
        elif tid in itos:
            decoded.append(itos[tid])
    return ' '.join(decoded)

def encode_tokens(tokens, stoi):
    """编码token列表"""
    encoded = []
    for token in tokens:
        if token in stoi:
            encoded.append(stoi[token])
    return encoded

def load_test_examples(data_path, meta, num_examples=500):
    """加载测试样例 - 正确版本"""
    stoi = meta['stoi']
    examples = []
    
    test_file = os.path.join(data_path, 'test.txt')
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return examples
    
    with open(test_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[:num_examples]:
        line = line.strip()
        if not line:
            continue
        
        tokens = line.split()
        if len(tokens) < 3:
            continue
        
        # 编码前3个token作为prompt
        prompt = encode_tokens(tokens[:3], stoi)
        
        if len(prompt) == 3:
            # 完整路径从第3个token开始（格式: source target source ... target）
            path = []
            for i in range(2, len(tokens)):
                if tokens[i].isdigit():
                    path.append(int(tokens[i]))
            examples.append((prompt, line, tuple(path)))
    
    return examples