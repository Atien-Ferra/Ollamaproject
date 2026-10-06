"""Retrieval + generation pipeline shared by chat.py and evaluate.py."""
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using only the context below, taken from the user's PDFs. "
     "If the answer is not in the context, say you don't know. Be concise.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def embeddings():
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_URL)


def vector_store():
    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings(),
        persist_directory=str(config.DB_DIR),
    )


class RAG:
    def __init__(self):
        self.store = vector_store()
        self.llm = ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)

    def retrieve(self, question: str):
        return self.store.similarity_search(question, k=config.TOP_K)

    def answer(self, question: str) -> dict:
        docs = self.retrieve(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        reply = (PROMPT | self.llm).invoke({"context": context, "question": question})
        return {
            "answer": reply.content,
            "contexts": [d.page_content for d in docs],
            "sources": [
                f"{d.metadata.get('source', '?')} p.{d.metadata.get('page', 0) + 1}" for d in docs
            ],
        }
