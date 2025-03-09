import configparser as cp
from src.LLM import DeepseekChat
config = cp.ConfigParser()
config.read('config.ini')

api_key = config.get('settings', 'api_key', fallback='no_key')
base_url = config.get('settings', 'base_url', fallback='no_url')
print(api_key, base_url)

# print(DeepseekChat().chat('You are copilot', 'Hi'))

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
docs = ReadFiles(file_path).get_content(max_token_len=500, cover_content=1)

