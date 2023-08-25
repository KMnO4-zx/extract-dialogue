from OpenAI_LLM import OpenAI_LLM
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import tiktoken
import json, time, jsonlines
from tqdm import tqdm
enc = tiktoken.get_encoding("cl100k_base")

schema = Object(
    id="script",
    description="Adapted from the novel into script",
    attributes=[
        Text(
            id="role",
            description="The character who is speaking",
        ),
        Text(
            id="dialogue",
            description="The dialogue spoken by the characters in the sentence",
        )
    ],
    examples=[
        (
            '''
            龙王说∶“再也没有比这更重的兵器了。”悟空不信，和龙王吵了起来，龙婆给龙王说∶“大禹治水时，测定海水深浅的神珍铁最近总是放光，就把这给他，管他能不能用，打发他走算了。”龙王听后告诉悟空∶“这宝物太重了，你自己去取吧！”
            ''',
            [
                {"role": "龙王", "dialogue": "再也没有比这更重的兵器了。"},
                {"role": "龙婆", "dialogue": "大禹治水时，测定海水深浅的神珍铁最近总是放光，就把这给他，管他能不能用，打发他走算了。”龙王听后告诉悟空∶“这宝物太重了，你自己去取吧！"},
            ],
        ),
        (
            '''
            悟空见八戒这么长时间不回来，就拔根毫毛变成自己，陪着师父和沙僧，真身驾云来到山凹里，见八戒和妖精正在交战，便高声叫道∶“八戒别慌，老孙来了！”八戒一听，来了精神，没几下，就把那群妖怪打败了。
            ''',
            [
                {"role": "悟空", "dialogue": "八戒别慌，老孙来了！"},
                # {"role": "克莱恩", "dialogue": "第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。"},
            ],
        )
    ],
    many=True,
)

def read_text(path):
    with open(path, mode='r', encoding='utf-8') as f:
        return f.read()

def save_data(data):
    filename = path.split('/')[-1].split('.')[0]
    with jsonlines.open(f"./output/{filename}.jsonl", mode='a') as f:
        f.write(data)


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


llm = OpenAI_LLM()
chain = create_extraction_chain(llm, schema)

# chunk_list = chunk_text_by_tokens(read_text('test.txt'), 1000)

def run(text):
    max_attempts = 3  # 最大尝试次数
    current_attempt = 1
    while current_attempt < max_attempts:
        try:
            response = chain.run(text)
        except Exception as e:
            print(e)
        else:
            break
        finally:
            print(f"第 {current_attempt} 次尝试完成。")
            current_attempt += 1

    if 'script' in response['data']:
        for item in response['data']['script']:
            print(item)
            save_data(json.dumps(item))
    else:
        passs


if __name__ == "__main__":
    path = '西游记白话文.txt'
    llm = OpenAI_LLM()
    chain = create_extraction_chain(llm, schema)
    chunk_list = get_chunk(read_text(path))
    for i in tqdm(range(len(chunk_list))):
        try:
            run(chunk_list[i])
        except Exception as e:
            print(e)