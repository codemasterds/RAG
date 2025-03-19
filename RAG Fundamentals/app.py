import os
from turtle import mode
from xml.dom.minidom import Document
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
#from chromadb.utils import embedding_functions
import chromadb.utils.embedding_functions as embedding_functions

#load enivronment variables from .env file
load_dotenv()

openai_key= os.getenv("OPENAI_API_KEY")
openai_ef= embedding_functions.OpenAIEmbeddingFunction(
    api_key= openai_key , model_name= "text-embedding-3-small"
)

#initalize chromedb persistentclient
chroma_client = chromadb.PersistentClient (
    path="chromadb-embeddings"
)
collection_name = "doc_qa_collection"
collection= chroma_client.get_or_create_collection(
    name=collection_name, embedding_function= openai_ef
)

#initalize openai obj
client= OpenAI(api_key=openai_key)

# #test response
# test_reponse = client.chat.completions.create(

#     model= "gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": "what is the capital city of Iowa state?"}
#     ]
# )

# print(test_reponse)


print("===loading docs===")
#function to load all the documents
def load_documents_from_directory(directory_path):
    
    documents=[]
    for filename in os.listdir(directory_path):
        if filename.endswith("txt"):
            with open(
                os.path.join (directory_path, filename), "r", encoding ="utf-8") as file:
                documents.append({"id":filename, "text":file.read()})
    return documents

#function to split the text into chunks
def split_text(text, chunk_size=1000, chunck_overlap=20):
    chunks=[]
    start=0
    while start<len(text):
        end= start+chunk_size
        #print("resr",text)
        chunks.append(text[start:end])
        start=end-chunck_overlap
    return chunks

directory="./news_articles"
documents= load_documents_from_directory(directory)
print(f"Loaded {len(documents)} docs")
  
#Split documents into chunks
print("===splitting into chunks===")
chunked_documents=[]
for document in documents:
    chunks= split_text(document['text'])
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id":f"{document['id']}_chunk{i+1}", "text":chunk})

print(f"Split documents into {len(chunked_documents)}")


def get_openai_embedding(text):
    response= client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding= response.data[0].embedding   
    return embedding

#generate the embeddings
print("==== Generating embeddings..====")
for doc in chunked_documents:
    doc["embedding"]= get_openai_embedding(doc["text"])

#upsert the embeddings in the chroma db database
print("===inserting chunks into chromadb")
for doc in chunked_documents:

    collection.upsert(ids=[doc["id"]], documents=[doc["text"]], embeddings=[doc["embedding"]])



def query_documents(question,n_results=2):
    
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Returning relevant chunks ====")
    return relevant_chunks

def generate_response(question, relevant_chunks):
    context= "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response= client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":"system",
                'content':prompt
            },
            {
                "role":"user",
                "content": question
            }

        ]
    )

    answer= response.choices[0].message
    return answer

question = "tell me about databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)