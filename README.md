# Local PDF RAG (LangChain + Ollama + ChromaDB + DeepEval)

Chat with your PDFs fully offline, then score the answers. It needs no paid API.

| File | Role |
|---|---|
| `ingest.py` | Loads PDFs from `data/`, splits them into chunks, embeds them with Ollama and stores them in ChromaDB (`chroma_db/`) |
| `chat.py` | Interactive Q&A over the index, with page-level sources |
| `evaluate.py` | DeepEval scoring (answer relevancy, faithfulness, contextual relevancy), with a local Ollama model as the judge |
| `rag.py` | The shared retrieve-then-generate pipeline |
| `config.py` | Models, chunk sizes and top-k. Each can be overridden with a `RAG_*` env var |

## Setup (one time)

```powershell
ollama pull llama3.2          # chat + judge model (~2 GB); embeddinggemma is already installed
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
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

Small local judges give noisier scores than GPT-4-class judges. If the results look erratic,
set `RAG_JUDGE_MODEL` to a larger model.
