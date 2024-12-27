import json
import ollama
import asyncio

async def run(context:str):
    messages = [
        {"role": "system", "content": '''你是一位善于从文本中提取标准关键信息的专家，例如以下情况：
          ```
          1.文本信息：[消防水量：根据《建筑设计防火规范》(GBJ 16-87，2001版),本工程电气楼、烧结室属高层厂房，戍类，室内、室外消防水量均为15L/s；]；
          识别标准：《建筑设计防火规范》(GBJ 16-87，2001版)；
          2.文本信息：[生活用水水质必须符合国家《生活饮用水卫生标准》(GB 5749-85),生产新水水质的悬浮物含量应小于30mg/L。]；
          识别标准：《生活饮用水卫生标准》(GB 5749-85)；
          3.文本信息：[1) 《通用用电设备配电设计规范》   GB50055-93
            2) 《建筑物防雷设计规范》         GB50057-94(2000版)
            3) 《建筑设计防火规范》           GBJ16-87(2001年版)]；
          识别标准：《通用用电设备配电设计规范》GB50055-93；《建筑物防雷设计规范》GB50057-94(2000版)；《建筑设计防火规范》GBJ16-87(2001年版)；
          ```
          你将根据输入的文本信息，识别出其中的标准，如果提供的文本信息中未识别到标准，请回答未检索到标准信息。
          .'''},
        {
        "role":'user',
        'content':context
        }
    ]
    client = ollama.AsyncClient()
    response = await client.chat(
        model='qwen2.5:14b',
        messages=messages,
        options={
            'temperature':0,
        },
    )
    print(response['message']['content'])
    
    
def standard_retrieve(context:str):
    asyncio.run(run(context))

if __name__ == '__main__':
    with open('C:\\Lee\\Projects\\ollama-python-example\\测试文件.md', 'r', encoding='utf-8') as file:
        content = file.read()
    chunk_size = 1000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    for chunk in chunks:
        standard_retrieve(chunk)
