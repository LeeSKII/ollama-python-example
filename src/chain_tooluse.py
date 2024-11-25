import json
import ollama
import asyncio
import datetime

def get_current_date():
    return json.dumps({'date':datetime.datetime.now().strftime("%Y-%m-%d")})

def get_flight_price(departure_city, arrival_city, departure_date):
    print(f'Getting flight price for {departure_city} to {arrival_city} on {departure_date}')
    flights = {
        '北京-上海': {'price': 320},
        '长沙-深圳': {'price': 620},
    }
    key = f'{departure_city}-{arrival_city}'
    return json.dumps(flights.get(key, {'error': 'Flight not found'}))

async def run(model:str):
    messages = [
            {
            "role":'user',
            'content':'查询今天的日期是2024-11-15号，使用工具查询北京至上海的机票价格是多少?'
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
                  'name': 'get_current_date',
                  'description': '获取今天的日期',
                },
            },
            {
                'type': 'function',
                'function': {
                'name': 'get_flight_price',
                'description': '根据日期获取机票的价格',
                'parameters': {
                    'type': 'object',
                    'properties': {
                    'departure_city': {
                        'type': 'string',
                        'description': '出发城市',
                    },
                    'arrival_city': {
                        'type': 'string',
                        'description': '到达城市',
                        },
                     'departure_date': {
                        'type': 'string',
                        'description': '出发日期',
                        },
                    },
                    'required': ['departure_city', 'arrival_city','departure_date'],
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
            'get_current_date': get_current_date,
            'get_flight_price':get_flight_price,
        }
        for tool in response['message']['tool_calls']:
            function_to_call = available_functions[tool['function']['name']]
            if(tool['function']['name'] == 'get_current_date'):
                function_response = function_to_call()
            elif(tool['function']['name'] == 'get_flight_price'):
                function_response = function_to_call(tool['function']['arguments']['departure_city'],tool['function']['arguments']['arrival_city'],tool['function']['arguments']['departure_date'])
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