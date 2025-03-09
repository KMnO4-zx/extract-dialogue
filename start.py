# utf-8
from src.extract import system_prompt
from src.schema import novel_schema
from src.LLM import DeepseekChat
from src.utils import ReadFiles
from tqdm import tqdm
import json
import configparser as cp

################################ 读取配置文件 ################################
config = cp.ConfigParser()
config.read('config.ini')
file_name = config.get('settings', 'file_name')

file_path = config.get('settings', 'file_path')
# 开始索引
start_idx = config.getint('progress', 'start_idx')
print(f"If this is not the first time executing the script, please check if start_idx {start_idx} has been updated in config.ini.")
user_input = input("If it has been updated, enter 'y' to continue: ").strip().lower()

# 检查输入
if user_input != "y":
    raise ValueError("Error: Execution aborted due to missing confirmation.")

print("Preparing to process.....")

# 最大token长度
max_token_len = config.getint('progress', 'max_token_len')
# 覆盖前面conver_content个单词
cover_content = config.getint('progress', 'cover_content')

################################ 文本对话提取 ################################

docs = ReadFiles(file_path).get_content(max_token_len=max_token_len, cover_content=cover_content)

sys_prompt = system_prompt(novel_schema)

model = DeepseekChat()

for i in tqdm(range(start_idx, len(docs)), total=len(docs), initial=start_idx):
    response = model.chat(sys_prompt, docs[i])
    try:
        
        response = json.loads(response)
        for item in response:
            with open(file_name, 'a', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(e)