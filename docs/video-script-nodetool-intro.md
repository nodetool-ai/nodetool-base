# NodeTool: 1-Minute Video Script

**Duration:** ~60 seconds  
**Target Audience:** Developers, AI enthusiasts, content creators  
**Tone:** Energetic, clear, inspiring

---

## SCRIPT

### [0:00 - 0:08] HOOK
**[Visual: Fast montage of AI workflows - images generating, text transforming, videos being created]**

> "What if you could build powerful AI workflows without writing endless boilerplate code? Meet NodeTool."

---

### [0:08 - 0:22] WHAT IS NODETOOL
**[Visual: NodeTool interface showing connected nodes forming a workflow graph]**

> "NodeTool lets you compose AI workflows as visual graphs. Connect nodes together like building blocks — each node does one thing really well, and together they create something powerful."

---

### [0:22 - 0:42] EXAMPLE WORKFLOW
**[Visual: Code/diagram showing the YouTube Thumbnail workflow being built step by step]**

> "Here's a real example: A YouTube Thumbnail Generator.
> 
> First, you input your video title and topic. 
> An AI Agent generates multiple creative thumbnail concepts.
> These feed into an image generator that creates the visuals.
> Then enhancement nodes boost contrast and brightness.
> Finally, text overlay nodes add your punchy headline.
> 
> One workflow. Multiple thumbnails. Ready for A/B testing."

---

### [0:42 - 0:52] CAPABILITIES OVERVIEW
**[Visual: Quick flashes of different node categories with icons]**

> "NodeTool comes with nodes for:
> - OpenAI, Claude, and Gemini integration
> - Image, audio, and video processing
> - Web scraping and data transformation
> - Vector databases for AI memory
> - And much more."

---

### [0:52 - 1:00] CALL TO ACTION
**[Visual: GitHub repo, terminal showing pip install, workflow running]**

> "Install with pip, connect your nodes, and let AI do the heavy lifting. NodeTool — compose AI workflows, visually.
> 
> Check out the GitHub repo to get started."

---

## PRODUCTION NOTES

### Key Messages
1. **Visual workflow composition** — AI as building blocks
2. **Practical use cases** — not abstract, real examples
3. **Rich ecosystem** — many nodes available out of the box

### Suggested B-Roll
- Node graph animations
- Code snippets (Python DSL examples)
- Generated outputs (thumbnails, images, text)
- Terminal showing workflow execution

### Example Code Snippet (for overlay)
```python
from nodetool.dsl.graph import create_graph
from nodetool.dsl.nodetool.agents import Agent
from nodetool.dsl.nodetool.image import TextToImage
from nodetool.dsl.nodetool.output import Output

# Connect nodes to build your workflow
agent = Agent(prompt="Generate thumbnail ideas...")
image = TextToImage(prompt=agent.out.text)
output = Output(name="thumbnail", value=image.output)
graph = create_graph(output)
```

### Hashtags & Keywords
`#AI #NoCode #Workflow #Automation #MachineLearning #NodeTool #Python #AITools`
