#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   schema.py
@Time    :   2024/06/16 07:54:09
@Author  :   不要葱姜蒜
@Version :   1.1
@Desc    :   None
'''

novel_schema = dict(
    task_description = 'Adapted from the novel into script', # 小说改编成剧本
    attributes = [
        dict(
            name='role', 
            description='The character who is speaking',
            type='String'
            ), # 角色
        dict(
            name='dialogue',
            description='The dialogue spoken by the characters in the sentence',
            type='String'
            ), # 对话
    ],
    example = [
        dict(
            text = """
            他下意识放轻了脚步，不制造出明显的噪音。
            刚登上二楼，他看见盥洗室的门突然打开，穿着旧布长裙的梅丽莎一副睡眼惺忪的模样出来。
            “你回来了……”梅丽莎还有些迷糊地揉了揉眼睛。
            克莱恩掩住嘴巴，打了个哈欠道：
            “是的，我需要一个美好的梦境，午餐之前都不要叫醒我。”
            梅丽莎“嗯”了一声，忽然想起什么似地说道：
            “我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。”
            """,
            script = [
                {"role": "梅丽莎", "dialogue": "你回来了……"},
                {"role": "克莱恩", "dialogue": "是的，我需要一个美好的梦境，午餐之前都不要叫醒我。"},
                {"role": "梅丽莎", "dialogue":"我和班森上午要去圣赛琳娜教堂做祈祷，参与弥撒，午餐可能会迟一点。"}
            ]
        )
    ]
)