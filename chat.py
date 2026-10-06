"""Interactive chat with your PDFs. Usage: python chat.py"""
import config
from rag import RAG


def main():
    rag = RAG()
    if rag.store._collection.count() == 0:
        raise SystemExit("The index is empty. Run `python ingest.py` first.")

    print(f"Chatting with {config.CHAT_MODEL}. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        result = rag.answer(question)
        print(f"\nAssistant: {result['answer']}")
        print("Sources: " + "; ".join(dict.fromkeys(result["sources"])) + "\n")


if __name__ == "__main__":
    main()
