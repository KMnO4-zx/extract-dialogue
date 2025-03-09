# utf-8
from src.extract import system_prompt
from src.schema import novel_schema
from src.LLM import DeepseekChat
from src.utils import ReadFiles
from tqdm import tqdm
import json
import configparser as cp
config = cp.ConfigParser()
config.read('config.ini')
file_name = config.get('settings', 'file_name')

file_path = config.get('settings', 'file_path')
print(file_path)

docs = ReadFiles(file_path).get_content(max_token_len=600, cover_content=50)

sys_prompt = system_prompt(novel_schema)

model = DeepseekChat()


for i in tqdm(range(len(docs))):
    response = model.chat(sys_prompt, docs[i])
    # print(f'第{i}次：', response)
    try:
        
        response = json.loads(response)
        # print(f'第{i}次 json.loads：', response)
        for item in response:
            with open(file_name, 'a', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(e)