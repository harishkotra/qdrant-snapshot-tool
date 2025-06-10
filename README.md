# Qdrant Snapshot Generator Tool

A simple web application that lets users upload `.txt`, `.md`, `.csv`, or `.pdf` files and generate a Qdrant-compatible embeddings snapshot using the `gte-Qwen2-1.5B-instruct-f16.gguf` model.

The resulting snapshot can be downloaded directly by the user for use in Gaia nodes or other RAG systems.

![image](https://github.com/user-attachments/assets/1a232323-da42-4317-b266-42694a06160a)

---

## 🔧 Features

✅ Upload multiple `.txt`, `.md`, `.csv`, `.pdf` files  
✅ Automatically converts PDFs to Markdown  
✅ Uses local GGUF model via WasmEdge to create embeddings  
✅ Stores vectors in Qdrant (embedded)  
✅ Creates a downloadable `.snapshot.tar.gz` file  
✅ Auto-cleanup of old snapshots (every hour)  
✅ Runs entirely in Docker – no setup needed  

---

## 🚀 How to Run

### 1. Prerequisites

Make sure you have these installed:

- [Docker](https://docs.docker.com/get-docker/) 
- [Docker Compose](https://docs.docker.com/compose/install/) 
- Download the gte-Qwen2-1.5B-instruct model inside the models folder using the following command:

```bash
curl -LO https://huggingface.co/gaianet/gte-Qwen2-1.5B-instruct-GGUF/resolve/main/gte-Qwen2-1.5B-instruct-f16.gguf
```

### 2. Prepare Models & WASM Tools

Place the following files in the correct folders before running Docker:

- `models/gte-Qwen2-1.5B-instruct-f16.gguf`  
- `wasm/markdown_embed.wasm`  
- `wasm/paragraph_embed.wasm`  
- `wasm/csv_embed.wasm`

You can get the `.wasm` tools from the [GaiaNet embedding-tools repo](https://github.com/GaiaNet-AI/embedding-tools). 

### 3. Build and Run

```bash
docker-compose up --build
```
Then open: 

```
👉 http://localhost:8000/  
```

## Project Structure

```
qdrant-snapshot-web/
├── app/
│   ├── main.py
│   └── static/
│       └── index.html
├── models/
│   └── gte-Qwen2-1.5B-instruct-f16.gguf
├── wasm/
│   ├── markdown_embed.wasm
│   ├── paragraph_embed.wasm
│   └── csv_embed.wasm
├── uploads/
├── qdrant_snapshots/
├── qdrant_storage/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🛠️ Development Setup (Optional) 

If you want to run locally without Docker: 

```bash
pip install -r requirements.txt
cd app/
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Make sure Qdrant is running separately: 

```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
