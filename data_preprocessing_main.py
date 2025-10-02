from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from vectorstore_creation import VectorStoreManager
import logging
import os

import time




class InvalidFileError(Exception):
    """Custom exception for invalid file types"""

    def __init__(self, message="Please provide only PDF files"):
        super().__init__(message)


class DocumentPreprocessor:

    def document_splitter(self, text):
        '''splits and chunks the text'''
        sentence_splitter = SpacyTextSplitter(
        chunk_size=2000,      # larger because sentences are preserved
        chunk_overlap=200
        )
        
        sentence_chunks = sentence_splitter.split_documents(text)

        # Step 3: Apply recursive fallback if sentence chunks are still too large
        final_chunks = []
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        for chunk in sentence_chunks:
            if len(chunk.page_content) > 1200:  # arbitrary threshold
                smaller_chunks = recursive_splitter.split_documents([chunk])
                final_chunks.extend(smaller_chunks)
            else:
                final_chunks.append(chunk)

        print(f"Original pages: {len(text)}")
        print(f"Sentence chunks: {len(sentence_chunks)}")
        print(f"Final chunks after hybrid split: {len(final_chunks)}")

        return final_chunks

    def load_and_chunk_file(self, file_path: str):
        """Load and chunk a single PDF file"""

        try:

            if not file_path.endswith(".pdf"):
                raise InvalidFileError("Please provide only PDF files")

            loader = PyPDFLoader(file_path)
            docs = loader.load()

            if not docs:
                print(f'File Path: {file_path}')
                with open('error_file.txt', 'w') as f:
                    f.write(file_path)
            


            chunks = self.document_splitter(docs)
            
            print(f"✅ Processed {file_path} → {len(chunks)} chunks")
            return chunks
        
        except Exception as e:
            logging.error(str(e))
            return str(e)


    def load_and_chunk_directory(self, folder_path: str):
        """Load and chunk all PDFs in a directory, one file at a time"""

        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
            
                if file_name.endswith(".pdf"):
                    yield self.load_and_chunk_file(file_path)  # returns chunks for this file only
                else:
                    InvalidFileError("⚠️ Please provide only PDF files")
        
        except Exception as e:
            print(str(e))






if __name__ == '__main__':
    start = time.time()
    d = DocumentPreprocessor()
    vectorstore = VectorStoreManager()
    
    chunks_generator = d.load_and_chunk_directory('../Data')
    vectorstore.add_chunks_from_generator(chunks_generator)

    print('=' * 10)
    print(f'Total processing time: {time.time() - start}')
    print('=' * 10)
