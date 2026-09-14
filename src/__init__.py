"""
src package
-----------
Core RAG building blocks, one responsibility per module:

    config.py           -> every tunable setting (paths, models, defaults)
    document_loader.py  -> detects file type and calls the right loader
    text_splitter.py    -> splits extracted text into overlapping chunks
    embeddings.py       -> turns text into vectors (sentence-transformers)
    vector_database.py  -> stores vectors + metadata and runs similarity search
    retriever.py        -> top-k retrieval, multi-query fusion (RRF)
    query_rewriter.py   -> manual (rule-based) and LLM-based query rewriting
    prompt.py           -> prompt templates (answering, rewriting, judging)
    llm.py              -> chat-completion client (Groq / Hugging Face router)
    rag_pipeline.py     -> glues everything into ask(question) -> answer + sources
"""
