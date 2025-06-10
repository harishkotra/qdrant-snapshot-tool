import os
import subprocess
import re
import pdfplumber
import requests
import tarfile
import time
import uuid
import threading
import glob
from datetime import datetime, timedelta
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient

app = FastAPI()

UPLOAD_DIR = "uploads"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
# IMPORTANT: Ensure this path is correct relative to where your FastAPI app runs.
# Make sure your 'models' directory and the GGUF model file exist.
MODEL_PATH = "models/gte-Qwen2-1.5B-instruct-f16.gguf" 

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Qdrant Client outside the request handler to reuse the connection
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# === Utility Functions ===

# This function is defined but not used in the current app logic,
# as WASM tools handle content splitting.
def split_markdown_by_heading(text):
    pattern = r'(^#+\s+.+?\n)'
    parts = re.split(pattern, text)
    sections = []
    current_section = ""
    for part in parts:
        if re.match(r'^#\s+', part):
            if current_section.strip():
                sections.append(current_section.strip())
                current_section = ""
            current_section = part.strip()
        else:
            current_section += "\n" + part.strip()
    if current_section.strip():
        sections.append(current_section.strip())
    return sections


def convert_pdf_to_markdown(pdf_path):
    """Converts a PDF file to a plain Markdown string."""
    md_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                md_text += page.extract_text() + "\n\n"
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {e}")
    return md_text

# This function is defined but not called anywhere in the current app.
def optimize_collection(collection_name):
    """Optimizes a Qdrant collection."""
    print("[Optimize] Optimizing collection...")
    client.update_collection(
        collection_name=collection_name,
        optimizer_config={"deleted_threshold": 0.2}
    )

# === API Endpoints ===

@app.get("/collections/{collection_name}")
def get_collection(collection_name: str):
    """Checks if a Qdrant collection exists."""
    try:
        # Attempt to get collection info. If it doesn't exist, QdrantClient raises an exception.
        info = client.get_collection(collection_name)
        # Using .dict() for compatibility across Pydantic versions
        return {"exists": True, "info": info.dict()} 
    except Exception:
        # Catch any exception, indicating the collection does not exist or cannot be accessed
        return JSONResponse(content={"exists": False}, status_code=404)


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str = Body(..., embed=True)
):
    """
    Uploads a file, converts it (if PDF), and embeds its content into a Qdrant collection
    using WasmEdge with a preloaded AI model.
    """
    allowed_types = ('.txt', '.md', '.csv', '.pdf')
    if not file.filename.lower().endswith(allowed_types):
        raise HTTPException(status_code=400, detail=f"Only {allowed_types} files are supported.")

    print(f"[Upload] Starting upload for collection '{collection_name}'")
    original_file_path = os.path.join(UPLOAD_DIR, file.filename)
    temp_input_path = original_file_path # Path used by wasmedge; might change for PDFs

    # Save the uploaded file
    try:
        with open(original_file_path, "wb") as f:
            f.write(await file.read())
        print(f"[Upload] Original file saved to {original_file_path}")
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Create collection only if it doesn't exist
    try:
        client.get_collection(collection_name)
        print(f"[Upload] Collection '{collection_name}' already exists.")
    except Exception:
        print(f"[Upload] Creating new collection '{collection_name}'.")
        try:
            client.create_collection(
                collection_name=collection_name,
                # Ensure vectors_config matches your model's embedding size and distance metric
                vectors_config={"size": 1536, "distance": "Cosine", "on_disk": True}
            )
        except Exception as e:
            # Clean up original file if collection creation fails
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            raise HTTPException(status_code=500, detail=f"Failed to create Qdrant collection: {e}")

    wasm_tool = None

    if file.filename.lower().endswith(".pdf"):
        print("[PDF] Converting PDF to Markdown...")
        try:
            md_text = convert_pdf_to_markdown(original_file_path)
            # Create a temporary Markdown file for wasmedge processing
            temp_input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}.md")
            with open(temp_input_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            wasm_tool = "wasm/markdown_embed.wasm"
        except Exception as e:
            # Clean up original file if PDF conversion fails
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            raise HTTPException(status_code=500, detail=f"PDF conversion failed: {e}")

    elif file.filename.lower().endswith(".md"):
        wasm_tool = "wasm/markdown_embed.wasm"
    elif file.filename.lower().endswith(".txt"):
        wasm_tool = "wasm/paragraph_embed.wasm"
    elif file.filename.lower().endswith(".csv"):
        wasm_tool = "wasm/csv_embed.wasm"
    
    if not wasm_tool:
        # This case should ideally be caught by allowed_types check, but good for robustness.
        # Clean up original file if no WASM tool is found (shouldn't happen with allowed_types)
        if os.path.exists(original_file_path):
            os.remove(original_file_path)
        raise HTTPException(status_code=400, detail="No suitable WASM tool found for this file type.")

    print(f"[Embed] Running {wasm_tool} with input {temp_input_path}")
    
    # Execute the wasmedge command to embed the content
    try:
        result = subprocess.run([
            "wasmedge",
            "--dir", ".:.", # Mount current directory for WASM access to files
            "--nn-preload", f"embedding:GGML:AUTO:{MODEL_PATH}", # Preload the AI model
            wasm_tool,
            "embedding", collection_name, "1536", temp_input_path, # Parameters for the WASM tool
            "--ctx_size", "8192" # Context size for the model
        ], capture_output=True, text=True, check=False) # check=False to handle errors manually

        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        if result.returncode != 0:
            error_message = f"Embedding process failed for {file.filename} (WASM tool: {wasm_tool}): {result.stderr or 'No stderr output'}"
            print(f"[Embed Error] {error_message}")
            raise HTTPException(status_code=500, detail=error_message)

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="WasmEdge runtime ('wasmedge' command) not found. Is WasmEdge installed and in PATH?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing embedding process: {e}")
    finally:
        # Clean up temporary files (if PDF) and the original uploaded file
        if temp_input_path != original_file_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            print(f"[Cleanup] Removed temporary converted file: {temp_input_path}")
        if os.path.exists(original_file_path): 
            os.remove(original_file_path)
            print(f"[Cleanup] Removed original uploaded file: {original_file_path}")

    # Verify points were added (optional but good for debugging and user feedback)
    try:
        point_count_response = client.count(collection_name=collection_name, exact=True)
        point_count = point_count_response.count
        print(f"[Upload] Collection '{collection_name}' now has {point_count} points after {file.filename} processing.")
    except Exception as e:
        print(f"[Upload Warning] Could not verify point count for collection {collection_name}: {e}")
    
    return {"collection_name": collection_name, "status": "success"}


@app.post("/snapshot/")
async def create_snapshot(collection_name: str = Body(..., embed=True)):
    """
    Requests Qdrant to create a snapshot for the specified collection,
    polls for its readiness, downloads it, and compresses it into a .tar.gz.
    """
    print("=== Starting snapshot creation ===")
    try:
        # Step 1: Request Qdrant to create snapshot
        print(f"[Snapshot] Requesting snapshot for '{collection_name}' from Qdrant.")
        snapshot_api_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/snapshots"
        
        response = requests.post(
            snapshot_api_url,
            headers={"Content-Type": "application/json"},
            json={} # Use json= parameter for automatic JSON serialization
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

        snapshot_data = response.json()
        print("[Snapshot] Snapshot generation request started. Qdrant response:", snapshot_data)

        snapshot_name = snapshot_data.get("result", {}).get("name")
        if not snapshot_name:
            raise Exception("No 'name' field found in Qdrant snapshot creation response.")

        print(f"[Snapshot] Expected snapshot name: {snapshot_name}")

        # Step 2: Poll for snapshot readiness by listing snapshots
        max_retries = 60 # Number of times to check
        retry_interval = 5 # seconds between checks
        snapshot_ready = False

        for i in range(max_retries):
            print(f"[Snapshot] Polling attempt {i+1}/{max_retries} to list snapshots for '{collection_name}'.")
            try:
                # GET request to list all snapshots for the collection
                list_response = requests.get(snapshot_api_url, timeout=10) 
                list_response.raise_for_status() # Raise an HTTPError for bad responses
                
                available_snapshots = list_response.json().get("result", [])
                
                # Check if the requested snapshot_name is present in the list
                if any(s.get("name") == snapshot_name for s in available_snapshots):
                    snapshot_ready = True
                    print("[Snapshot] Snapshot is listed and ready!")
                    break
            except requests.exceptions.RequestException as e:
                print(f"[Snapshot] Request to list snapshots failed (attempt {i+1}): {str(e)}")
            except Exception as e:
                print(f"[Snapshot] Error processing snapshot list response (attempt {i+1}): {str(e)}")
            
            time.sleep(retry_interval)

        if not snapshot_ready:
            raise HTTPException(
                status_code=504, 
                detail="Timed out waiting for snapshot to be created and listed by Qdrant. "
                       "Check Qdrant logs for errors."
            )

        # Step 3: Download the snapshot file
        print("[Snapshot] Downloading snapshot...")
        # Construct the correct URL for downloading a specific snapshot
        snapshot_download_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection_name}/snapshots/{snapshot_name}"
        
        # Define path for the raw downloaded snapshot file
        snapshot_raw_path = os.path.join(UPLOAD_DIR, snapshot_name) 

        # Stream the download to handle potentially large files
        download_response = requests.get(snapshot_download_url, stream=True, timeout=300) # Increased timeout for large downloads
        download_response.raise_for_status() # Raise an HTTPError for bad responses

        # Save the downloaded snapshot file chunk by chunk
        with open(snapshot_raw_path, "wb") as f:
            for chunk in download_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"[Snapshot] Raw snapshot downloaded to {snapshot_raw_path}")

        # Step 4: Compress into .tar.gz
        # The frontend expects {collection_name}.snapshot.tar.gz for download
        final_tar_path = os.path.join(UPLOAD_DIR, f"{collection_name}.snapshot.tar.gz")
        print(f"[Snapshot] Creating tar.gz archive at {final_tar_path}")
        
        with tarfile.open(final_tar_path, "w:gz") as tar:
            # Add the raw snapshot file to the archive using its original name (snapshot_name)
            tar.add(snapshot_raw_path, arcname=snapshot_name)

        print("[Snapshot] ✅ Snapshot successfully created and compressed.")

        # Step 5: Clean up the intermediate raw .snapshot file
        try:
            os.remove(snapshot_raw_path)
            print(f"[Snapshot] Cleaned up raw snapshot file: {snapshot_raw_path}")
        except Exception as e:
            print(f"[Snapshot] Could not remove raw snapshot file {snapshot_raw_path}: {e}")

        return {"snapshot_file": final_tar_path}

    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. Please ensure Qdrant is running and accessible. Error: {e}")
    except requests.exceptions.RequestException as e:
        # Catch any other requests-related errors (e.g., HTTP errors, timeouts)
        raise HTTPException(status_code=500, detail=f"Qdrant API error during snapshot creation: {e}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"[Snapshot] ❌ Unhandled error creating snapshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during snapshot creation: {str(e)}")


@app.get("/download/{collection_name}")
async def download_snapshot(collection_name: str):
    """
    Provides the compressed snapshot file for download.
    """
    tar_path = os.path.join(UPLOAD_DIR, f"{collection_name}.snapshot.tar.gz")

    if not os.path.exists(tar_path):
        raise HTTPException(status_code=404, detail="Snapshot not found or not yet created.")

    return FileResponse(
        path=tar_path, 
        filename=os.path.basename(tar_path), 
        media_type='application/gzip',
        # Set content disposition to attachment for forced download
        headers={"Content-Disposition": f"attachment; filename=\"{collection_name}.snapshot.tar.gz\""}
    )


# === Auto-Cleanup Logic ===

def auto_cleanup_snapshots():
    """
    Periodically cleans up old snapshot files and tar.gz archives from the UPLOAD_DIR.
    Files older than 24 hours are removed.
    """
    while True:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Cleanup] Running auto-cleanup...")
        now = datetime.now()
        # Keep snapshots only for 24 hours by default
        cleanup_time_delta = timedelta(hours=24) 
        cutoff = now - cleanup_time_delta
        
        # Search for all snapshot files (*.snapshot) and tar.gz files (*.tar.gz)
        # within the UPLOAD_DIR
        tar_files = glob.glob(os.path.join(UPLOAD_DIR, "*.tar.gz"))
        snapshot_files = glob.glob(os.path.join(UPLOAD_DIR, "*.snapshot")) 

        removed_count = 0
        for path in tar_files + snapshot_files:
            try:
                # Get last modification time of the file
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(path))
                if file_mod_time < cutoff:
                    os.remove(path)
                    print(f"[Cleanup] Removed old file: {path}")
                    removed_count += 1
            except Exception as e:
                print(f"[Cleanup] Failed to remove {path}: {str(e)}")
        print(f"[Cleanup] Removed {removed_count} old files.")
        time.sleep(3600)  # Run cleanup every hour (3600 seconds)

# Start cleanup thread as a daemon so it exits gracefully with the main application
cleanup_thread = threading.Thread(target=auto_cleanup_snapshots, daemon=True)
cleanup_thread.start()


# === Serve Static Files ===

# This mounts the 'static' directory to the root '/', serving index.html
# and any other static assets (like CSS) from there.
app.mount("/", StaticFiles(directory="static", html=True), name="static")