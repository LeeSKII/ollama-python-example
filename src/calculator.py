import json
import ollama
import asyncio

def add(first,second):
    # 需要返回json格式才能被ollama解析
    return json.dumps({'result':first+second})

def divide(first,second):
    return json.dumps({'result':first/second})

def sqrt(number):
    return json.dumps({'result':round(number**0.5,2)})

async def run(model:str):
    messages = [
            {
            "role":'user',
            'content':'6/2的结果再加上100的结果开平方根是多少?'
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
                'name': 'add',
                'description': '获取两个数相加的结果',
                'parameters': {
                    'type': 'object',
                    'properties': {
                    'first': {
                        'type': 'number',
                        'description': '第一个数字',
                    },
                    'second': {
                        'type': 'number',
                        'description': '第二个数字',
                        },
                    },
                    'required': ['first', 'second'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                'name': 'divide',
                'description': '获取两个数相除的结果',
                'parameters': {
                    'type': 'object',
                    'properties': {
                    'first': {
                        'type': 'number',
                        'description': '被除数',
                    },
                    'second': {
                        'type': 'number',
                        'description': '除数',
                        },
                    },
                    'required': ['first', 'second'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                'name': 'sqrt',
                'description': '获取一个数的平方根',
                'parameters': {
                    'type': 'object',
                    'properties': {
                    'number': {
                        'type': 'number',
                        'description': '数字',
                    },
                    
                    },
                    'required': ['number'],
                    },
                },
            },
        ],
    )

    messages.append(response['message'])
    print(response)

    # Process function calls made by the model
    if response['message'].get('tool_calls'):
        available_functions = {
            'add': add,
            'divide':divide,
            'sqrt': sqrt,
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            if(tool['function']['name'] == 'add' or tool['function']['name'] == 'divide'):
                function_response = function_to_call(tool['function']['arguments']['first'], tool['function']['arguments']['second'])
            else:
                function_response = function_to_call(tool['function']['arguments']['number'])
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
    print(f'messages={messages}')  
    
asyncio.run(run('qwen2.5:7b'))