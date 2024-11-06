import json
import ollama
import asyncio

def get_travel_info(user_name:str, session_id:str):
    if(user_name == 'Lee' and session_id == 'asdssad'):
        # 注意需要设置不编码的中文字符，否则大模型可能会解析成别的字符导致信息错误
        return json.dumps({'result':f'用户：{user_name}，session_id：{session_id}，最近一条出差信息为：出差时间：2024年9月23日-2024年9月30日， 出发地：上海，目的地：北京，出差事由：参加全国钢铁冶金污染治理会议。'},ensure_ascii=False)
    else:
        return '您好，您没有权限访问该功能，请联系管理员。'

async def run(model:str):
    messages = [
        {
            "role":'system',
            "content":'你是一个提供查询出差信息的助手，当前登录用户是：Lee，权限为：passenger，session_id为：asdssad。'
        },
        {
            "role":'user',
            "content":'请查询关于我的详细出差信息。'
        }
    ]
    client = ollama.AsyncClient()
    response = await client.chat(
        model=model,
        messages=messages,
        options={
            'temperature':0,
        },
        tools=[
            {
                'type': 'function',
                'function': {
                'name': 'get_travel_info',
                'description': '获取用户最近的出差信息',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_name': {
                            'type': 'string',
                            'description': '用户名',
                        },
                        'session_id': {
                            'type': 'string',
                            'description': 'session_id',
                        },
                    },
                    'required': ['user_name', 'session_id'],
                },
                },
            },
        ]
    )
    
    messages.append(response['message'])
    
    # Check if the model decided to use the provided function
    if not response['message'].get('tool_calls'):
        print("The model didn't use the function. Its response was:")
        print(response['message']['content'])

    # Process function calls made by the model
    if response['message'].get('tool_calls'):
        available_functions = {
            'get_travel_info': get_travel_info,
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            function_response = function_to_call(tool['function']['arguments']['user_name'], tool['function']['arguments']['session_id'])
            print(f"The model used the {tool['function']['name']} function with arguments {tool['function']['arguments']}. Its response was:")
            print(function_response)
            # Add function response to the conversation
            messages.append(
                {
                    'role':'tool',
                    'content':function_response
                }
            )
        
        # second API call:Get final response from the model
        final_response = await client.chat(model=model,messages=messages)
        print(f'final response:{final_response}')
    print(response)
    # print(f'messages:{messages}')
    
if __name__ == '__main__':
    asyncio.run(run('qwen2.5:7b'))