import ollama
import chromadb

documents = [
  "lee是一名著名运动员",
  "dell电脑TX675使用intel core i7处理器并且拥有16gb内存，1tb硬盘",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )
  
  
  # an example prompt
prompt = "请告知我一些关于火箭运动的情况"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="mxbai-embed-large"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=1
)
data = results['documents'][0][0]

# TODO: 这里应该对data的distance进行判断，如果距离超过一定阈值，则不采用该数据

print(f"The most relevant document to the prompt '{prompt}' is: {results}")

# generate a response combining the prompt and data we retrieved in step 2
output = ollama.generate(
  model="qwen2.5:7b",
  prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
)

print(output['response'])