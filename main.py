from OpenAI_LLM import OpenAI_LLM
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
import tiktoken
import json
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
            他下意识放轻了脚步，不制造出明显的噪音。
            刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
            “你回来了……”梅丽莎还有些迷糊地揉了揉眼睛。
            克莱恩掩住嘴巴，打了个哈欠道：
            “是的，我需要一个美好的梦境，午餐之前都不要叫醒我。”
            梅丽莎“嗯”了一声，忽然想起什么似地说道：
            “我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。”
            ''',
            [
                {"role": "梅丽莎", "dialogue": "你回来了……"},
                {"role": "克莱恩", "dialogue": "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"},
                {"role": "梅丽莎", "dialogue":"我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"}
            ],
        ),
        (
            '''
            “太感谢您了！‘愚者’先生您真是太慷慨了！”奥黛丽欣喜地回应道。
            她为自己刚才想用金钱购买消息的庸俗忏悔了三秒。
            克莱恩停止手指的敲动，语气平淡地描述道：
            “第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。”
            我不知不觉竟然用上了队长的口吻……克莱恩的嘴角下意识就翘了起来。
            ''',
            [
                {"role": "奥黛丽", "dialogue": "太感谢您了！‘愚者’先生您真是太慷慨了！"},
                {"role": "克莱恩", "dialogue": "第一个常识，非凡特性不灭定律，非凡特性不会毁灭，不会减少，只是从一个事物转移到另一个事物。"},
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
    with open(f"./output/{filename}.jsonl", mode='a', encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False,indent=4)+'\n')


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
    try:
        response = chain.run(text)
    except Exception as e:
        print(e)

    if 'script' in response['data']:
        for item in response['data']['script']:
            print(item)
            save_data(item)
    else:
        passs


if __name__ == "__main__":
    path = 'test.txt'
    llm = OpenAI_LLM()
    chain = create_extraction_chain(llm, schema)
    chunk_list = get_chunk(read_text(path))
    for i in tqdm(range(len(chunk_list))):
        try:
            run(chunk_list[i])
        except Exception as e:
            print(e)