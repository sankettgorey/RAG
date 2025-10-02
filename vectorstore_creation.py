from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


class VectorStoreManager:
    def __init__(self, persist_dir="./supreme_court_verdicts"):

        self.persist_dir = persist_dir
        
        self.embedding_function = OllamaEmbeddings(model="nomic-embed-text:latest")

        self.vectorstore = Chroma(persist_directory=self.persist_dir, 
                                  embedding_function=self.embedding_function,
                                  collection_name='final_verdicts')

    def add_chunks_from_generator(self, chunks_generator):
        """
        Accepts a generator that yields chunks for each file.
        Adds them to the vectorstore one file at a time.
        """
        total = 0
        for file_chunks in chunks_generator:
            self.vectorstore.add_documents(file_chunks)
            # self.vectorstore.persist()
            total += len(file_chunks)
            print(f"✅ Stored {len(file_chunks)} chunks from one file. Total stored: {total}")

        print(f"🎉 All chunks from generator stored. Total chunks: {total}")