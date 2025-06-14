"""
实验运行脚本 - 自动运行多个实验
使用方法: python run_experiment.py
"""
import subprocess
import os
import json
from datetime import datetime

def run_single_experiment(name, config):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"开始实验: {name}")
    print(f"配置: {config}")
    print(f"{'='*60}\n")
    
    # 构建命令
    cmd = ["python", "train.py"]
    for key, value in config.items():
        cmd.extend([f"--{key}", str(value)])
    
    # 运行
    print(f"运行命令: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    print(f"\n实验 '{name}' 完成!")

def main():
    """主函数"""
    print("相变实验批量运行脚本")
    print("="*60)
    
    # 定义实验配置
    experiments = {
        "baseline_200k": {
            "max_iters": 200000,
            "test_interval": 1000,
            "num_eval_batches": 10
        },
        
        "early_stop_100k": {
            "max_iters": 100000,
            "test_interval": 1000,
            "num_eval_batches": 10
        },
        
        "dense_monitor": {
            "max_iters": 150000,
            "test_interval": 500,  # 更频繁的测试
            "num_eval_batches": 10
        }
    }
    
    print("可用的实验:")
    for i, (name, config) in enumerate(experiments.items(), 1):
        print(f"  {i}. {name} (max_iters={config['max_iters']})")
    print(f"  {len(experiments)+1}. 运行所有实验")
    
    choice = input("\n请选择要运行的实验 (输入数字): ").strip()
    
    selected_experiments = {}
    if choice == str(len(experiments)+1):
        selected_experiments = experiments
    else:
        try:
            idx = int(choice) - 1
            name = list(experiments.keys())[idx]
            selected_experiments = {name: experiments[name]}
        except:
            print("无效选择，运行baseline实验")
            selected_experiments = {"baseline_200k": experiments["baseline_200k"]}
    
    # 记录开始时间
    start_time = datetime.now()
    print(f"\n开始时间: {start_time}")
    
    # 运行选定的实验
    for name, config in selected_experiments.items():
        run_single_experiment(name, config)
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n所有实验完成!")
    print(f"总用时: {duration}")
    print(f"\n现在可以运行分析脚本查看结果:")
    print(f"  python analyze_phase.py")

if __name__ == "__main__":
    main()