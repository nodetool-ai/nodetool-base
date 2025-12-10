import json
import uuid
import datetime
import sys
import os
from typing import Any, Dict
from unittest.mock import MagicMock

# Mock huggingface_hub to avoid dependency issues in this environment
sys.modules["huggingface_hub"] = MagicMock()
sys.modules["huggingface_hub.inference"] = MagicMock()
sys.modules["huggingface_hub.inference._providers"] = MagicMock()

from pydantic import BaseModel

# Define a dummy ModelInfo class for Pydantic compatibility
class ModelInfo(BaseModel):
    pass

sys.modules["huggingface_hub"].ModelInfo = ModelInfo
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.fernet"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["cryptography.hazmat.primitives"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf"] = MagicMock()
sys.modules["cryptography.hazmat.primitives.kdf.pbkdf2"] = MagicMock()
sys.modules["cryptography.hazmat.backends"] = MagicMock()
sys.modules["boto3"] = MagicMock()
sys.modules["botocore"] = MagicMock()
sys.modules["botocore.exceptions"] = MagicMock()
sys.modules["keyring"] = MagicMock()
sys.modules["keyring.errors"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()
sys.modules["chromadb.utils"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions.ollama_embedding_function"] = MagicMock()
sys.modules["chromadb.utils.embedding_functions.sentence_transformer_embedding_function"] = MagicMock()
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["langchain_text_splitters"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.client"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

# Add the workspace root to python path to allow imports
sys.path.append("/Users/mg/workspace/nodetool-base/examples")
sys.path.append("/Users/mg/workspace/nodetool-core/src")
sys.path.append("/Users/mg/workspace/nodetool-base/src")

# Import the graphs from the example files
# We use importlib to import by path or module name
import importlib.util

def load_graph_from_file(filepath: str):
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return module.graph

EXAMPLES = [
    {
        "filename": "ai_workflows_on_reddit.py",
        "json_name": "AI Workflows on Reddit.json",
        "name": "AI Workflows on Reddit",
        "description": "Find examples of AI workflows on Reddit and compile a markdown report.",
        "tags": ["reddit", "research", "ai"]
    },
    {
        "filename": "wikipedia_agent_example.py",
        "json_name": "Wikipedia Agent.json",
        "name": "Wikipedia Agent",
        "description": "Wikipedia style research and documentation agent.",
        "tags": ["wikipedia", "research", "writing"]
    },
    {
        "filename": "learning_path_generator.py",
        "json_name": "Learning Path Generator.json",
        "name": "Learning Path Generator",
        "description": "Create a comprehensive learning path for a topic.",
        "tags": ["education", "planning"]
    },
    {
        "filename": "instagram_scraper_task.py",
        "json_name": "Instagram Scraper.json",
        "name": "Instagram Scraper",
        "description": "Analyze Instagram trends for a specific topic.",
        "tags": ["instagram", "social-media", "trends"]
    },
    {
        "filename": "reddit_scraper_agent.py",
        "json_name": "Reddit Scraper.json",
        "name": "Reddit Scraper",
        "description": "Analyze a subreddit for specific issues.",
        "tags": ["reddit", "research", "analysis"]
    },
    {
        "filename": "test_hackernews_agent.py",
        "json_name": "Hacker News Agent.json",
        "name": "Hacker News Agent",
        "description": "Scrape and analyze the front page of Hacker News.",
        "tags": ["hackernews", "news", "analysis"]
    },
    {
        "filename": "chromadb_research_agent.py",
        "json_name": "ChromaDB Research Agent.json",
        "name": "ChromaDB Research Agent",
        "description": "Query a Chroma collection of papers.",
        "tags": ["chromadb", "rag", "research"]
    },
    {
        "filename": "product_hunt_ai_extractor_agent.py",
        "json_name": "Product Hunt AI Extractor.json",
        "name": "Product Hunt AI Extractor",
        "description": "Identify AI products from Product Hunt leaderboards.",
        "tags": ["product-hunt", "ai", "extraction"]
    }
]

SOURCE_DIR = "/Users/mg/workspace/nodetool-base/examples"
TARGET_DIR = "/Users/mg/workspace/nodetool-base/src/nodetool/examples/nodetool-base"

def generate_json():
    for example in EXAMPLES:
        print(f"Processing {example['filename']}...")
        filepath = os.path.join(SOURCE_DIR, example['filename'])
        try:
            graph = load_graph_from_file(filepath)
            
            # Construct the full JSON object
            now = datetime.datetime.now().isoformat()
            full_json = {
                "id": str(uuid.uuid4()),
                "access": "public",
                "created_at": now,
                "updated_at": now,
                "name": example["name"],
                "tool_name": None,
                "description": example["description"],
                "tags": example["tags"],
                "thumbnail": None,
                "thumbnail_url": None,
                "graph": graph.model_dump(),
                "input_schema": None,
                "output_schema": None,
                "settings": None,
                "package_name": "nodetool-base",
                "path": None,
                "run_mode": None,
                "required_providers": None,
                "required_models": None
            }
            
            target_path = os.path.join(TARGET_DIR, example['json_name'])
            with open(target_path, "w") as f:
                json.dump(full_json, f, indent=2)
            print(f"Generated {target_path}")
            
        except Exception as e:
            print(f"Failed to process {example['filename']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_json()
