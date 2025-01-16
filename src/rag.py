import ollama
import chromadb

documents = [
  "印度空间研究组织（ISRO）1月16日宣布，航天器对接成功完成，继美国、俄罗斯和中国后，成为第四个达到这一重大里程碑的国家。印度极地卫星运载火箭于去年12月30日搭载两枚用于“空间对接试验”（SpaDeX）任务的卫星顺利升空，原计划于今年1月7日左右尝试完成交会对接。",
  "当地时间1月15日，卡塔尔首相兼外交大臣穆罕默德在多哈宣布，以色列与巴勒斯坦伊斯兰抵抗运动（哈马斯）就加沙地带停火和被扣押人员交换达成协议，该协议将于1月19日生效，哈马斯将释放33名被扣押人员，以换取以色列释放巴勒斯坦被扣押人员。",
  "1 月 15 日，在国新办举行“中国经济高质量发展成效”系列新闻发布会上，商务部市场运行和消费促进司司长李刚表示，商务部本周将陆续印发 2025 年加力支持汽车、家电、家装和电动自行车以及手机等数码产品的购新补贴实施细则。",
  "个人消费者购买单件销售价格不超过 6000 元的手机、平板、智能手表（手环）3 类数码产品，可享受购新补贴。每人每类可补贴 1 件，每件补贴比例为减去生产、流通环节及移动运营商所有优惠后最终销售价格的 15%，每件最高不超过 500 元。各地将根据该方案细化实施细则，1 月 20 日起全国实施"
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
prompt = "现在买手机有优惠吗？"

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