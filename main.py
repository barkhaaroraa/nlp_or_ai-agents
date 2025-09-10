import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'


dataset = []
with open('cat-facts.txt', 'r') as file:
  dataset = file.readlines()
  print(f'Loaded {len(dataset)} entries')
print(dataset[1])

VECTOR_DB = []

def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f" added chunk {i+1}/{len(dataset)} to dataset")

def cosine_sim(a,b):
  dot_product= sum([x*y for x,y in zip(a,b)])
  norm_a=sum([x**2 for x in a])**0.5
  norm_b=sum([x**2 for x in b])**0.5
  return dot_product/(norm_a*norm_b)

def retrieve(query, top_n=10):
  query_embed=ollama.embed(model=EMBEDDING_MODEL,input=query)['embeddings'][0]
  similarities=[]

  for chunk,embedding in VECTOR_DB:
    similarity=cosine_sim(query_embed,embedding)
    similarities.append((chunk,similarity))
    #sort
  similarities.sort(key=lambda x: x[1],reverse=True)
  return similarities[:top_n]

while True:
  inputq = input("\nAsk (or 'q' to quit): ")
  
  if inputq.lower() == 'q':
    print("Goodbye!")
    break
  
  retrieved_answer = retrieve(inputq)
  
  print("\nRetrieving relevant context...")
  
  for chunk, similarity in retrieved_answer:
    print(f'(similarity: {similarity:.2f}) {chunk}')
  
  instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_answer])}
'''
  
  stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
      {'role': 'system', 'content': instruction_prompt},
      {'role': 'user', 'content': inputq},
    ],
    stream=True,
  )
  
  print('\nChatbot response: ')
  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
  print("\n" + "="*50)

