# ollama-qwen-demo
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text2QAPipeline_LocalAPI.py
---------------------------
这是一个接入本地 API (Local API) 的版本。
适用于你已经在本地启动了 Qwen3 (千问) 的 API 服务（例如使用 Ollama）。

前提条件：
1. 本地已安装 Ollama 并运行: ollama serve
2. 已拉取模型: ollama pull qwen3:4b
"""

import argparse
import os
import json
import time
import requests  # 需要安装: pip install requests
from pathlib import Path

# =============================================================================
# 1. 真实 API 调用组件 (Real API Component)
# =============================================================================

class RealLLMServing:
    """真实调用本地 API 的服务类"""
    def __init__(self, model_name, api_base_url, api_key="EMPTY"):
        self.model_name = model_name
        self.api_url = f"{api_base_url}/chat/completions"
        self.api_key = api_key
        print(f"[LLM] Connecting to Local API: {self.api_url}")
        print(f"[LLM] Target Model: {self.model_name}")

    def chat(self, prompt):
        """发送请求给本地 API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024
        }

        try:
            print(f"     (Sending request to {self.model_name}...)")
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=600)
            response.raise_for_status()
            
            result = response.json()
            # 兼容 OpenAI 格式的返回
            answer = result['choices'][0]['message']['content']
            return answer
            
        except Exception as e:
            print(f"     [Error] API Call Failed: {e}")
            return f"[Error] Failed to get response: {str(e)}"

# =============================================================================
# 2. 模拟/简化的 Pipeline 组件 (保持流程通畅)
# =============================================================================

class FileStorage:
    def __init__(self, first_entry_file_name, cache_path, file_name_prefix, cache_type):
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        # 我们可以创建一个真实的输入文件用于测试
        if not os.path.exists(first_entry_file_name):
            with open(first_entry_file_name, 'w', encoding='utf-8') as f:
                # 写入一些真实文本用于测试 Qwen 的能力
                f.write(json.dumps({"text_id": "1", "content": "DeepSeek is an AI company based in China. It has developed the DeepSeek-V3 and R1 models."}) + "\n")
        print(f"[Storage] Initialized. Input file: {first_entry_file_name}")

    def step(self):
        return self

class KBCChunkGeneratorBatch:
    def __init__(self, split_method, chunk_size, tokenizer_name):
        pass
    def run(self, storage):
        print("  -> Splitting text... (Using dummy logic for demo)")
        # 实际这里应该读文件，为了演示简单，我们假设内存里已经有一段文本
        self.current_text = "DeepSeek is an AI company based in China. It has developed the DeepSeek-V3 and R1 models."

class KBCTextCleanerBatch:
    def __init__(self, llm_serving, lang):
        self.llm = llm_serving
    def run(self, storage):
        print("  -> Cleaning text using Qwen...")
        prompt = f"Please clean and correct the following text (keep it concise):\n\n{KBCChunkGeneratorBatch.current_text if hasattr(KBCChunkGeneratorBatch, 'current_text') else 'Hello World'}"
        response = self.llm.chat(prompt)
        print(f"     [Qwen Response]: {response[:100]}...")
        self.cleaned_text = response

class KBCMultiHopQAGeneratorBatch:
    def __init__(self, llm_serving, lang):
        self.llm = llm_serving
    def run(self, storage):
        print("  -> Generating QA pairs using Qwen...")
        # 使用上一步清洗过的文本，或者默认文本
        context = "DeepSeek is an AI company based in China."
        prompt = f"Based on the following text, generate 2 question-answer pairs in JSON format:\n\n{context}"
        response = self.llm.chat(prompt)
        print(f"     [Qwen Response]: {response[:100]}...")
        self.qa_result = response

class QAExtractor:
    def __init__(self, qa_key, output_json_file):
        self.output_file = output_json_file
    def run(self, storage, input_key, output_key):
        print("  -> Saving results...")
        # 这里简单把最后一次模型生成的结果存进去
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({"raw_llm_output": "See console log for Qwen output"}, indent=2))
        print(f"  -> Done. Check {output_path}")

# =============================================================================
# 3. 主流程
# =============================================================================

class Text2QAPipeline:
    def __init__(self, cache_base="./"):
        cache_path = Path(cache_base).resolve()
        
        # === 配置你的本地 API 地址 ===
        # Ollama 默认地址
        LOCAL_API_URL = "http://localhost:11434/v1" 
        
        # === 配置你的模型名称 ===
        LOCAL_MODEL_NAME = "qwen3:4b"

        self.storage = FileStorage(
            first_entry_file_name=str(cache_path / ".cache" / "gpu" / "text_input.jsonl"),
            cache_path=str(cache_path / ".cache" / "gpu"),
            file_name_prefix="text2qa_step",
            cache_type="json",
        )

        self.text_splitting_step = KBCChunkGeneratorBatch("token", 512, "qwen")

        # 替换为真实 API 服务
        self.llm_serving = RealLLMServing(
            model_name=LOCAL_MODEL_NAME,
            api_base_url=LOCAL_API_URL,
            api_key="EMPTY" # 本地服务通常不需要 Key
        )

        self.knowledge_cleaning_step = KBCTextCleanerBatch(self.llm_serving, "en")
        self.qa_generation_step = KBCMultiHopQAGeneratorBatch(self.llm_serving, "en")
        
        self.extract_format_qa = QAExtractor(
            qa_key="qa_pairs",
            output_json_file=str(cache_path / ".cache" / "data" / "qa_local.json"),
        )

    def forward(self):
        print("\n=== Pipeline Start (Local Qwen3:4b) ===")
        self.text_splitting_step.run(self.storage)
        self.knowledge_cleaning_step.run(self.storage)
        self.qa_generation_step.run(self.storage)
        self.extract_format_qa.run(self.storage, "", "")
        print("\n=== Pipeline Completed ===")

def main():
    # 检查是否安装了 requests
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library is missing. Please run: pip install requests")
        return

    model = Text2QAPipeline(cache_base="./")
    model.forward()

if __name__ == "__main__":
    main()
