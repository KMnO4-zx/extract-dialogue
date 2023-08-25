import json

dialogue_list = []

with open('./output/西游记白话文.jsonl', mode='r', encoding='utf-8') as file:
    for line in file:
        print(json.loads(line))
