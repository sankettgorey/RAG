
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA


# persist_directory = "./chroma_db1"
embedding_function = OllamaEmbeddings(model = 'nomic-embed-text:latest')


# Load embeddings (same as before)
# Load existing DB
vectorstore = Chroma(
    persist_directory="./supreme_court_verdicts",   # directory you used earlier
    embedding_function=embedding_function,
    collection_name="final_verdicts"     # must match what you used earlier
)

print("✅ Vectorstore loaded with", vectorstore._collection.count(), "documents")


retriever = vectorstore.as_retriever(
    search_type="mmr",   # or "mmr" (Max Marginal Relevance)
    search_kwargs={"k": 5, "lambda_mult": 0.3}      # number of chunks to retrieve
)


llm = ChatOllama(model = 'mistral:7b-instruct')
# llm = ChatOllama(model = 'qwen2.5:7b-instruct')


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True   # so you can see which chunks were used
)


query = "What is the case related to 60321-1971___jonew__judis__9229? can you tell me why this case was filed"
# query = "give me the list of all cases you have. givem me only names"
# query = "why municipal corporation of delhi filed the case? what was the case all about?"
result = qa_chain.invoke({"query": query})

print("\nAnswer:\n", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("---")
    print(doc.page_content[:300])  # preview source
