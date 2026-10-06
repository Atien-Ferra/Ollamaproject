# Local PDF RAG (LangChain + Ollama + ChromaDB + DeepEval)

Chat with your PDFs fully offline, then score the answers. It needs no paid API.

| File | Role |
|---|---|
| `ingest.py` | Loads PDFs from `data/`, splits them into chunks, embeds them with Ollama and stores them in ChromaDB (`chroma_db/`) |
| `chat.py` | Interactive Q&A over the index, with page-level sources |
| `evaluate.py` | DeepEval scoring (answer relevancy, faithfulness, contextual relevancy), with a local Ollama model as the judge |
| `rag.py` | The shared retrieve-then-generate pipeline |
| `config.py` | Models, chunk sizes, top-k and file paths. Each can be overridden with a `RAG_*` env var |

## Setup (one time)

Requires [Ollama](https://ollama.com) and Python 3.10+.

```powershell
ollama pull llama3.2          # chat model (~2 GB)
ollama pull embeddinggemma    # embedding model (~620 MB)
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Use

```powershell
# 1. Put PDFs in data\
.\.venv\Scripts\python ingest.py          # add --reset to rebuild from scratch
.\.venv\Scripts\python chat.py
# 2. Write questions about your PDFs in eval_questions.json
.\.venv\Scripts\python evaluate.py
```

To use a different model, for example Llama 3 8B: `$env:RAG_CHAT_MODEL="llama3"`. If you change
`RAG_EMBED_MODEL`, re-run `ingest.py --reset`.

## About the evaluation scores

The scores from `evaluate.py` are only as good as the judge model. By default the judge is the
same model as the chat model, `llama3.2` (3B). That model is too small to judge reliably. In
testing it gave correct, word-for-word answers an Answer Relevancy score of 0.00, and its stated
reasons contradicted its own scores. Use a larger judge for meaningful numbers, for example:

```powershell
ollama pull qwen2.5:14b
$env:RAG_JUDGE_MODEL="qwen2.5:14b"
.\.venv\Scripts\python evaluate.py
```

Evaluation runs one request at a time with DeepEval's timeouts turned off, because a local judge
can't handle parallel requests. Expect it to take a few minutes per question on a small model.
