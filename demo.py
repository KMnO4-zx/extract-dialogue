from openai import OpenAI
import tiktoken
import json, time
from tqdm import tqdm
from schema import schema
import pprint
import json
import pprint
enc = tiktoken.get_encoding("cl100k_base")

client = OpenAI(api_key="", base_url="https://api.deepseek.com/v1")



PROMPT = '''Your goal is to extract structured information from the user's input that matches the form described below. When extracting information please make sure it matches the type information exactly. Do not add any attributes that do not appear in the schema shown below.

```TypeScript

script: Array< // Adapted from the novel into script
    role: string // The character who is speaking
    dialogue: string // The dialogue spoken by the characters in the sentence
>
```

Please output the extracted information in JSON format in Excel dialect. 
Do NOT add any clarifying information. Output MUST follow the schema above. Do NOT add any additional columns that do not appear in the schema.

Input: {emaple_input}
Output: {example_output}

Input: {user_input}
Output:
'''

def get_prompt(schema, user_input):
    return PROMPT.format(
        emaple_input=schema['example']['input'],
        example_output=schema['example']['output'],
        user_input=user_input,
    )

def read_text(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return f.read()

def get_chunk(text):
    """
    text: str
    return: chunk_text
    """
    max_token_len = 600
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行

    for line in lines:
        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len
    
    if curr_chunk:
        chunk_text.append(curr_chunk)
    
    return chunk_text

def get_completion(prompt, model="deepseek-chat"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # 模型输出的温度系数，控制输出的随机程度
    )
    return response.choices[0].message.content


tetx = '''
    “我也这么觉得！”克莱恩深有同感地附和。

    和塔利姆相视一笑后，他起身前往地下靶场练枪练非凡能力，快到中午的时候，才返回一楼，直奔自助餐厅。

    他之前已经注意到，今天的限量供应是，红酒煎鹅肝，并搭配有苹果片和浸入了黄油的面包。

    取好食物，克莱恩端着餐盘，走向塔利姆在的那张桌子，而此时此刻，那里除了塔利姆，还有另外一位他的熟人，同样作为担保，介绍他进入俱乐部的外科医生艾伦.克瑞斯。

    刚放好餐盘，还未来得及坐下，克莱恩突然发现那位知名外科医生的椅子旁边靠了根拐杖。

    “艾伦，怎么了？”他关心地问了一句。

    个子高瘦，长相冷淡，戴着副金丝边眼镜的艾伦轻拍了下右腿道：

    “不，不要提，这真是太倒霉了！我从楼梯上摔了下去，出现了较为严重的骨裂，只能打石膏做固定。”

    “真是不够走运啊。”克莱恩附和着叹息了一声，并切了块鹅肝，蘸汁塞入口中，那刚一接触就会融化般的感觉让脂肪的芳香不断外扩，刺激着每一个味蕾。

    “我已经不走运很长一段时间了。”艾伦举手推了下镜框，顺势揉了揉额角。

    他随即望向克莱恩，又看了看塔利姆，犹豫着问道：

    “莫里亚蒂先生，你有没有，有没有……”
'''


prompt = get_prompt(schema, tetx)
response = get_completion(prompt)
print(type(response))
response = json.loads(response)
pprint.pp(response)

