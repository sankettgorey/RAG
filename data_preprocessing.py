from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SpacyTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


files = ['./Data/-0___jonew__judis__6080.pdf', './Data/-0___jonew__judis__5201.pdf', './Data/-0___jonew__judis__4924.pdf']

for file in files:
    loader = PyPDFLoader(file)

    text = loader.load()



    # Step 2: Sentence-aware split
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



    embedding_function = OllamaEmbeddings(model = 'nomic-embed-text:latest')


    try:
        # Load existing Chroma DB
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_function,
            collection_name="legal_chunks"
        )

        # Add new docs
        vectorstore.add_documents(final_chunks)

        # Save / persist
        vectorstore.persist()

        print("✅ Added", len(final_chunks), "new chunks into Chroma")
    except Exception as e:
        print(str(e))

