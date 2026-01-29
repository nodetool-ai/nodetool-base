import json
import uuid
import datetime
import sys
import os
from unittest.mock import MagicMock
from pydantic import BaseModel

sys.path.append(os.path.dirname(__file__))

from ai_workflows_on_reddit import graph as ai_workflows_on_reddit_graph
from wikipedia_agent_example import graph as wikipedia_agent_example_graph
from learning_path_generator import graph as learning_path_generator_graph
from instagram_scraper_task import graph as instagram_scraper_task_graph
from reddit_scraper_agent import graph as reddit_scraper_agent_graph
from test_hackernews_agent import graph as test_hackernews_agent_graph
from chromadb_research_agent import graph as chromadb_research_agent_graph
from product_hunt_ai_extractor_agent import graph as product_hunt_ai_extractor_agent_graph
from concept_art_iteration_board import graph as concept_art_iteration_board_graph
from album_cover_creator import graph as album_cover_creator_graph
from brand_asset_generator import graph as brand_asset_generator_graph
from product_mockup_generator import graph as product_mockup_generator_graph


EXAMPLES = [
    {
        "filename": "concept_art_iteration_board.py",
        "json_name": "Concept Art Iteration Board.json",
        "name": "Concept Art Iteration Board",
        "description": "Create a comprehensive learning path for a topic.",
        "tags": ["concept-art", "planning"],
        "graph": concept_art_iteration_board_graph
    },

    {
        "filename": "ai_workflows_on_reddit.py",
        "json_name": "AI Workflows on Reddit.json",
        "name": "AI Workflows on Reddit",
        "description": "Find examples of AI workflows on Reddit and compile a markdown report.",
        "tags": ["reddit", "research", "ai"],
        "graph": ai_workflows_on_reddit_graph
    },
    {
        "filename": "wikipedia_agent_example.py",
        "json_name": "Wikipedia Agent.json",
        "name": "Wikipedia Agent",
        "description": "Wikipedia style research and documentation agent.",
        "tags": ["wikipedia", "research", "writing"],
        "graph": wikipedia_agent_example_graph
    },
    {
        "filename": "learning_path_generator.py",
        "json_name": "Learning Path Generator.json",
        "name": "Learning Path Generator",
        "description": "Create a comprehensive learning path for a topic.",
        "tags": ["education", "planning"],
        "graph": learning_path_generator_graph
    },
    {
        "filename": "instagram_scraper_task.py",
        "json_name": "Instagram Scraper.json",
        "name": "Instagram Scraper",
        "description": "Analyze Instagram trends for a specific topic.",
        "tags": ["instagram", "social-media", "trends"],
        "graph": instagram_scraper_task_graph
    },
    {
        "filename": "reddit_scraper_agent.py",
        "json_name": "Reddit Scraper.json",
        "name": "Reddit Scraper",
        "description": "Analyze a subreddit for specific issues.",
        "tags": ["reddit", "research", "analysis"],
        "graph": reddit_scraper_agent_graph
    },
    {
        "filename": "test_hackernews_agent.py",
        "json_name": "Hacker News Agent.json",
        "name": "Hacker News Agent",
        "description": "Scrape and analyze the front page of Hacker News.",
        "tags": ["hackernews", "news", "analysis"],
        "graph": test_hackernews_agent_graph
    },
    {
        "filename": "chromadb_research_agent.py",
        "json_name": "ChromaDB Research Agent.json",
        "name": "ChromaDB Research Agent",
        "description": "Query a Chroma collection of papers.",
        "tags": ["chromadb", "rag", "research"],
        "graph": chromadb_research_agent_graph
    },
    {
        "filename": "product_hunt_ai_extractor_agent.py",
        "json_name": "Product Hunt AI Extractor.json",
        "name": "Product Hunt AI Extractor",
        "description": "Identify AI products from Product Hunt leaderboards.",
        "tags": ["product-hunt", "ai", "extraction"],
        "graph": product_hunt_ai_extractor_agent_graph
    },
    {
        "filename": "album_cover_creator.py",
        "json_name": "Album Cover Creator.json",
        "name": "Album Cover Creator",
        "description": "Create an album cover for a given song.",
        "tags": ["album-cover", "art", "design"],
        "graph": album_cover_creator_graph
    },
    {
        "filename": "brand_asset_generator.py",
        "json_name": "Brand Asset Generator.json",
        "name": "Brand Asset Generator",
        "description": "Generate brand assets for a given brand.",
        "tags": ["brand-asset", "branding", "design"],
        "graph": brand_asset_generator_graph
    },
    {
        "filename": "product_mockup_generator.py",
        "json_name": "Product Mockup Generator.json",
        "name": "Product Mockup Generator",
        "description": "Generate product mockups for a given product.",
        "tags": ["product-mockup", "mockup", "design"],
        "graph": product_mockup_generator_graph
    }
]

TARGET_DIR: str = os.path.join(os.path.dirname(__file__), "..", "src", "nodetool", "examples", "nodetool-base")

def generate_json():
    for example in EXAMPLES:
        print(f"Processing {example['filename']}...")
        try:
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
                "graph": example["graph"].model_dump(),
                "input_schema": None,
                "output_schema": None,
                "settings": None,
                "package_name": "nodetool-base",
                "path": None,
                "run_mode": None,
                "required_providers": None,
                "required_models": None
            }
            
            target_path = os.path.join(TARGET_DIR, str(example['json_name']))
            with open(target_path, "w") as f:
                json.dump(full_json, f, indent=2)
            print(f"Generated {target_path}")
            
        except Exception as e:
            print(f"Failed to process {example['filename']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    generate_json()
