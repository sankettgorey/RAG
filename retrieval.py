
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA


persist_directory = "./chroma_db1"
embedding_function = OllamaEmbeddings(model = 'nomic-embed-text:latest')

# Load embeddings (same as before)

# Load existing DB
vectorstore = Chroma(
    persist_directory="./chroma_db1",   # directory you used earlier
    embedding_function=embedding_function,
    collection_name="legal_chunks"     # must match what you used earlier
)

print("✅ Vectorstore loaded with", vectorstore._collection.count(), "documents")


retriever = vectorstore.as_retriever(
    search_type="similarity",   # or "mmr" (Max Marginal Relevance)
    search_kwargs={"k": 5}      # number of chunks to retrieve
)


llm = ChatOllama(model = 'mistral:7b-instruct')
# llm = ChatOllama(model = 'qwen2.5:7b-instruct')



qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True   # so you can see which chunks were used
)



query = "What is the main judgment in case related to STATE OF KARNATAKA?"
result = qa_chain.invoke({"query": query})

print("\nAnswer:\n", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("---")
    print(doc.page_content[:300])  # preview source




