from fastapi.background import P
from ollama import chat
from pydantic import BaseModel

class Country(BaseModel):
  name: str
  capital: str
  languages: list[str]

# stream = chat(
#     model='qwen2.5:7b',
#     messages=[{'role': 'user', 'content': '告诉我一些关于加拿大的信息。'}],
#     format=Country.model_json_schema(),
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

response = chat(
  messages=[
    {
      'role': 'user',
      'content': '告诉我一些关于加拿大的信息.',
    }
  ],
  model='qwen2.5:7b',
  format=Country.model_json_schema(),
)

country = Country.model_validate_json(response.message.content)
print(country)