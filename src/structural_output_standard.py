from fastapi.background import P
from ollama import chat
from pydantic import BaseModel

class Standard(BaseModel):
    name: str
    code: str

class StandardList(BaseModel):
    standards: list[Standard]

def standard_retrieve(context: str):
    response = chat(
        messages = [
            {"role": "system", "content": '''你是一位善于从文本中提取标准关键信息的专家，例如以下情况：
                ```
                1.文本信息：消防水量：根据《建筑设计防火规范》(GBJ 16-87，2001版),本工程电气楼、烧结室属高层厂房，戍类，室内、室外消防水量均为15L/s；；
                识别：[{name:建筑设计防火规范;code:GBJ 16-87，2001版}]；
                2.文本信息：生活用水水质必须符合国家《生活饮用水卫生标准》(GB 5749-85),生产新水水质的悬浮物含量应小于30mg/L。；
                识别：[{name:生活饮用水卫生标准;code:GB 5749-85}]；
                3.文本信息：
                1) 《通用用电设备配电设计规范》   GB50055-93
                2) 《建筑物防雷设计规范》         GB50057-94(2000版)
                3) 《建筑设计防火规范》           GBJ16-87(2001年版)；
                识别：[{name:通用用电设备配电设计规范;code:GB50055-93}；{name:建筑物防雷设计规范;code:GB50057-94(2000版)}；{name:建筑设计防火规范;code:GBJ16-87(2001年版)}]；
                4.文本信息：GB 5749-2006 《室外排水设计规范》；
                识别：[{name:室外排水设计规范；;code:GB 5749-2006}]；
                ```
                你将根据输入的文本信息，识别出其中的标准，如果提供的文本信息中未识别到标准，请直接返回null，不需要做其它多余的回答。
                .'''},
            {
            "role":'user',
            'content':context
            }
        ],
        model='qwen2.5:7b',
        format=StandardList.model_json_schema(),
    )

    list = StandardList.model_validate_json(response.message.content)
    if len(list.standards)>0:
        print(list.standards)
    
if __name__ == '__main__':
    with open('C:\\Lee\\Projects\\ollama-python-example\\测试文件.md', 'r', encoding='utf-8') as file:
        content = file.read()
    chunk_size = 500
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    for chunk in chunks:
        standard_retrieve(chunk)
    
    # text = '''
    #         # 给排水

    #     ## 设计范围

    #     本设计为烧结烟气净化系统红线范围以内的生产及消防给水；生产排水；雨水排放。

    #     本工程所需能源水介质包括工业水、消防给水，所有能源水介质由业主提供，接口位置为项目红线范围以外1米处。

    #     ## 设计采用的规范和标准

    #     GB 50013 -2006《室外给水设计规范》

    #     GB 50014 -2006《室外排水设计规范》（2014年版）

    #     GB 50015-2003《建筑给水排水设计规范》（2009版）

    #     GB 50016 -2006《建筑设计防火规范》
    # '''
    # standard_retrieve(text)