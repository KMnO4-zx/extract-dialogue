from openai import OpenAI
import json
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

SYSTEM_PROMPT = """
Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.
{TypeScript}
Please output the extracted information in JSON format in Excel dialect. 

Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: 
{Input}
Output:
{Output}
"""

TYPE_SCRIPT = """
```TypeScript
    script: Array<Dict( // {task_description}
    {attributes}
    )>
```
"""

def get_typescript(schema, type_script_str):
    script = schema['attributes']
    script_str = ',\n    '.join([f"{s['name']}: {s['type']} // {s['description']} " for s in script])
    type_script_str = type_script_str.format(task_description=schema['task_description'], attributes=script_str)
    return type_script_str

def system_prompt(schema):
    return SYSTEM_PROMPT.format(TypeScript=get_typescript(schema, TYPE_SCRIPT), Input=schema['example'][0]['text'], Output=json.dumps(schema['example'][0]['script'], indent=4, ensure_ascii=False))

