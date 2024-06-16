# Extract Dialogue

>***本仓库只为`huanhuan-chat`泛化版的一部分内容（文本对话抽取），欢迎大家给`huanhuan-chat`仓库star！本仓库的最大贡献就是为泛化的Character AI提供了从小说中建立数据集的功能。***
>
>`huanhuan-chat: https://github.com/KMnO4-zx/huanhuan-chat.git`

## Show

`repo`：https://github.com/KMnO4-zx/extract-dialogue.git

本项目利用`chatgpt`从小说中提取对话集，提取的样本中包括`role`，`dialogue`，比如以下的形式：

```json
{
    "role": "艾伦",
    "dialogue": "不，不要提，这真是太倒霉了！我从楼梯上摔了下去，出现了较为严重的骨裂，只能打石膏做固定。"
}
{
    "role": "克莱恩",
    "dialogue": "真是不够走运啊。"
}
```

## QuickStart

- 克隆仓库并切换目录：`git clone https://github.com/KMnO4-zx/extract-dialogue.git `，`cd extract-dialogue`

- 安装依赖：`pip install -r requirements.txt`
- 在当前目录创建`.env`文件，并填入`DEEPSEEK_API`。
- 把你要提取的小说或文本，放到当前目录，在`example.py`中修改`path`。
- ***强烈建议您结合要提取的小说修改`schema.py`中的`schema`示例。在下面的部分中有详细介绍`schema`。***

- 运行`example.py`，`python example.py`

结果如下所示：

```json
{"role": "克莱恩", "dialogue": "在帮警察们调查那起连环杀人案，虽然不一定能有收获，但赏金足够诱人，而且，和警察部门建立良好的关系对我们私家侦探来说非常重要。"}
{"role": "塔利姆", "dialogue": "这果然是大侦探忙碌的事情。"}
{"role": "塔利姆", "dialogue": "莫里亚蒂先生，我能请教一个问题吗？"}
{"role": "克莱恩", "dialogue": "这单免费，还有，叫我夏洛克就行了。"}
{"role": "塔利姆", "dialogue": "我有个朋友，爱上了不该爱的人，这种情况该怎么处理？"}
{"role": "塔利姆", "dialogue": "莫里亚蒂先生，我能请教一个问题吗？"}
{"role": "克莱恩", "dialogue": "这单免费，还有，叫我夏洛克就行了。"}
{"role": "塔利姆", "dialogue": "我有个朋友，爱上了不该爱的人，这种情况该怎么处理？"}
{"role": "克莱恩", "dialogue": "我唯一的建议是，不要犯法。"}
{"role": "克莱恩", "dialogue": "首先，我们要弄清楚‘不该’是源于什么？双方的家庭之间有仇恨关系？"}
{"role": "塔利姆", "dialogue": "不，这不是《罗密欧与朱丽叶》的故事！"}
```


## Introduction

```python
from extract import system_prompt
from schema import novel_schema
from LLM import DeepseekChat
from utils import ReadFiles
from tqdm import tqdm
import json

file_path = './data/test.txt'
docs = ReadFiles(file_path).get_content(max_token_len=500, cover_content=0)

sys_prompt = system_prompt(novel_schema)

model = DeepseekChat()

file_name = file_path.split('/')[-1].split('.')[0]

for i in tqdm(range(len(docs))):
    response = model.chat(sys_prompt, docs[i])
    try:
        response = json.loads(response)
        for item in response:
            with open(f'{file_name}.jsonl', 'a', encoding='utf-8') as f:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(e)
```

## 参考

【1】https://eyurtsev.github.io/kor/index.html#

【2】https://zhuanlan.zhihu.com/p/646948797