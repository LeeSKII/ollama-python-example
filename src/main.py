import json
import ollama

def get_flight_times(departure: str, arrival: str) -> str:
    flights = {
        'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
        'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
        'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
        'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
        'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
        'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
    }

    key = f'{departure}-{arrival}'.upper()
    return json.dumps(flights.get(key, {'error': 'Flight not found'}))
  

response = ollama.chat(
  model='llama3.2',
  messages=[
    {
    "role":'user',
    'content':'What is the flight time from New York (nyc) to Los Angeles (LAX)?'
    }
  ],
 tools=[
        {
            'type': 'function',
            'function': {
            'name': 'get_flight_times',
            'description': 'Get the flight times between two cities',
            'parameters': {
                'type': 'object',
                'properties': {
                'departure': {
                    'type': 'string',
                    'description': 'The departure city (airport code)',
                },
                'arrival': {
                    'type': 'string',
                    'description': 'The arrival city (airport code)',
                    },
                },
                'required': ['departure', 'arrival'],
            },
            },
        },
    ],
)

print(response)

# Check if the model decided to use the provided function
if not response['message'].get('tool_calls'):
    print("The model didn't use the function. Its response was:")
    print(response['message']['content'])

  # Process function calls made by the model
if response['message'].get('tool_calls'):
    available_functions = {
        'get_flight_times': get_flight_times,
    }
    for tool in response['message']['tool_calls']:
        function_to_call = available_functions[tool['function']['name']]
        function_response = function_to_call(tool['function']['arguments']['departure'], tool['function']['arguments']['arrival'])
        # Add function response to the conversation
        print(f"The model used the {tool['function']['name']} function with arguments {tool['function']['arguments']}. Its response was:")
        print(function_response)