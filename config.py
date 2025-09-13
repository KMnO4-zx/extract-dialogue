#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件：统一管理对话提取器的所有配置
"""

import os
from typing import Dict, List
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

class ModelPlatform:
    """模型平台配置类"""
    
    # 支持的模型平台
    PLATFORMS = {
        'deepseek': {
            'api_key_env': 'DEEPSEEK_API',
            'base_url_env': 'DEEPSEEK_BASE_URL',
            'default_base_url': 'https://api.deepseek.com',
            'default_model': 'deepseek-chat',
            'description': 'DeepSeek AI平台'
        },
        'openai': {
            'api_key_env': 'OPENAI_API_KEY',
            'base_url_env': 'OPENAI_BASE_URL',
            'default_base_url': 'https://api.openai.com/v1',
            'default_model': 'gpt-3.5-turbo',
            'description': 'OpenAI官方平台'
        },
        'siliconflow': {
            'api_key_env': 'SILICONFLOW_API_KEY',
            'base_url_env': 'SILICONFLOW_BASE_URL',
            'default_base_url': 'https://api.siliconflow.cn/v1',
            'default_model': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
            'description': 'SiliconFlow AI平台'
        },
        'moonshot': {
            'api_key_env': 'MOONSHOT_API_KEY',
            'base_url_env': 'MOONSHOT_BASE_URL',
            'default_base_url': 'https://api.moonshot.cn/v1',
            'default_model': 'kimi-k2-0905-preview',
            'description': '月之暗面Kimi平台'
        },
        'custom': {
            'api_key_env': 'CUSTOM_API_KEY',
            'base_url_env': 'CUSTOM_BASE_URL',
            'default_base_url': 'https://your-custom-endpoint.com/v1',
            'default_model': 'custom-model',
            'description': '自定义API端点'
        }
    }
    
    @classmethod
    def get_platform_config(cls, platform: str) -> dict:
        """获取指定平台的配置"""
        if platform not in cls.PLATFORMS:
            raise ValueError(f"不支持的平台: {platform}。支持的平台: {list(cls.PLATFORMS.keys())}")
        
        return cls.PLATFORMS[platform]
    
    @classmethod
    def list_platforms(cls) -> dict:
        """列出所有支持的平台"""
        return {name: config['description'] for name, config in cls.PLATFORMS.items()}

class Config:
    """配置管理类"""
    
    # 默认平台
    DEFAULT_PLATFORM = 'siliconflow'
    
    # 从环境变量获取平台配置
    CURRENT_PLATFORM = os.getenv('LLM_PLATFORM', DEFAULT_PLATFORM)
    
    # 通用配置
    TEMPERATURE = 0.6
    MAX_RETRIES = 3
    RETRY_DELAY = 2
    
    # 多线程配置
    MAX_WORKERS = 8  # 最大并发线程数（默认8个）
    QUEUE_TIMEOUT = 30  # 队列超时时间（秒）
    BATCH_SIZE = 10  # 批处理大小
    
    # 默认行为配置
    DEFAULT_SHOW_STATS = True  # 默认显示统计信息
    DEFAULT_CONCURRENT = True  # 默认使用多线程并发处理
    DEFAULT_SORT_OUTPUT = False  # 默认完成后按chunk_id排序输出
    DEFAULT_SAVE_CHUNK_TEXT = False  # 默认不保存原始chunk文本
    
    @classmethod
    def get_current_platform_config(cls) -> dict:
        """获取当前平台的配置"""
        platform_config = ModelPlatform.get_platform_config(cls.CURRENT_PLATFORM)
        
        api_key = os.getenv(platform_config['api_key_env'])
        base_url = os.getenv(platform_config['base_url_env'], platform_config['default_base_url'])
        model_name = os.getenv(f"{cls.CURRENT_PLATFORM.upper()}_MODEL_NAME", platform_config['default_model'])
        
        if not api_key:
            available_platforms = []
            for platform_name, config in ModelPlatform.PLATFORMS.items():
                if os.getenv(config['api_key_env']):
                    available_platforms.append(platform_name)
            
            if available_platforms:
                raise ValueError(
                    f"平台 {cls.CURRENT_PLATFORM} 的API密钥未设置。\n"
                    f"请设置环境变量 {platform_config['api_key_env']}，\n"
                    f"或使用其他已配置的平台: {', '.join(available_platforms)}\n"
                    f"可通过设置 LLM_PLATFORM 环境变量切换平台"
                )
            else:
                env_list = [f'  - {config["api_key_env"]}' for config in ModelPlatform.PLATFORMS.values()]
                raise ValueError(
                    f"未找到任何已配置的API密钥。\n"
                    f"请至少设置一个平台的API密钥:\n"
                    f"{chr(10).join(env_list)}"
                )
        
        return {
            'platform': cls.CURRENT_PLATFORM,
            'api_key': api_key,
            'base_url': base_url,
            'model_name': model_name,
            'description': platform_config['description']
        }
    
    @classmethod
    def set_platform(cls, platform: str):
        """设置当前平台"""
        if platform not in ModelPlatform.PLATFORMS:
            raise ValueError(f"不支持的平台: {platform}")
        cls.CURRENT_PLATFORM = platform
    
    # 文本处理配置
    MAX_TOKEN_LEN = 500
    COVER_CONTENT = 50
    ENCODING = "cl100k_base"
    
    # 输出配置
    OUTPUT_ENCODING = 'utf-8'
    OUTPUT_FORMAT = 'jsonl'
    
    # Chunk-ID 配置
    INCLUDE_CHUNK_ID = True  # 是否在输出中包含chunk-id
    SAVE_CHUNK_TEXT = True  # 是否保存原始chunk文本（默认开启）
    BUFFER_BEFORE_WRITE = True  # 是否缓冲结果后按顺序写入
    
    # 默认对话提取模式
    DEFAULT_SCHEMA = {
        'task_description': '从小说中提取角色对话',
        'attributes': [
            {
                'name': 'role',
                'description': '说话的角色名称',
                'type': 'String'
            },
            {
                'name': 'dialogue', 
                'description': '角色说的对话内容',
                'type': 'String'
            }
        ],
        'example': [
            {
                'text': '''
                他下意识放轻了脚步，不制造出明显的噪音。
                刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
                "你回来了……"梅丽莎还有些迷糊地揉了揉眼睛。
                克莱恩掩住嘴巴，打了个哈欠道：
                "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"
                梅丽莎"嗯"了一声，忽然想起什么似地说道：
                "我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"
                ''',
                'script': [
                    {"role": "梅丽莎", "dialogue": "你回来了……"},
                    {"role": "克莱恩", "dialogue": "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"},
                    {"role": "梅丽莎", "dialogue": "我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"}
                ]
            }
        ]
    }
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "dialogue_extractor.log"
    
    # 进度恢复配置
    CACHE_DIR = ".cache"
    PROGRESS_FILE = "progress.json"
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        try:
            cls.get_current_platform_config()
        except ValueError as e:
            errors.append(str(e))
        
        if cls.MAX_TOKEN_LEN <= cls.COVER_CONTENT:
            errors.append("MAX_TOKEN_LEN必须大于COVER_CONTENT")
            
        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            errors.append("TEMPERATURE必须在0-2之间")
            
        return errors
    
    @classmethod
    def get_system_prompt_template(cls) -> str:
        """获取系统提示模板"""
        return """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.
{TypeScript}
Please output the extracted information in JSON format. 

Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: 
{Input}
Output:
{Output}
"""

    @classmethod
    def get_typescript_template(cls) -> str:
        """获取TypeScript模板"""
        return """
```TypeScript
    script: Array<Dict( // {task_description}
    {attributes}
    )>
```
"""