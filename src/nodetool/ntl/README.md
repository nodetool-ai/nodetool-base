# NTL - NodeTool Language

**NTL** is a simple, elegant, and token-efficient text format for defining nodetool workflows.

## Design Goals

- **Simple**: Easy to read and write by humans and LLMs
- **Elegant**: Clean syntax with minimal punctuation
- **Token-efficient**: Compact representation for LLM context windows
- **Expressive**: Support all workflow features (nodes, edges, metadata)

## Syntax Overview

### Workflow Metadata

Workflow metadata is defined at the top of the file using key-value pairs prefixed with `@`:

```ntl
@name "Image Enhancement Pipeline"
@description "Enhance images with sharpening and contrast"
@tags image, enhancement, start
```

### Node Declarations

Nodes are declared using their ID, type, and optional properties:

```ntl
node_id: NodeType
  property1 = value
  property2 = "string value"
```

**Node Types** use dot notation matching the nodetool namespace:
- `nodetool.input.ImageInput`
- `lib.pillow.enhance.Sharpen`
- `nodetool.output.Output`

**Properties** support multiple value types:
- Strings: `"hello world"`
- Numbers: `42`, `3.14`, `-10`
- Booleans: `true`, `false`
- References (connections): `@other_node.output`
- Objects: `{type: "image", uri: "https://..."}`
- Lists: `[1, 2, 3]` or `["a", "b", "c"]`

### Connections/Edges

Connections are expressed using the `->` operator:

```ntl
source_node.output_handle -> target_node.input_handle
```

Or inline in property assignments using `@` reference:

```ntl
sharpen: lib.pillow.enhance.Sharpen
  image = @image_input.output
```

### Comments

Single-line comments start with `#`:

```ntl
# This is a comment
input: nodetool.input.ImageInput  # Inline comment
```

Multi-line comments use `/* ... */`:

```ntl
/*
  This workflow enhances images
  using Pillow filters.
*/
```

## Complete Example

```ntl
# Image Enhancement Workflow
@name "Image Enhance"
@description "Improve image quality with basic enhancement tools"
@tags image, start

# Input node
image_input: nodetool.input.ImageInput
  name = "image"
  value = {
    type: "image",
    uri: "https://example.com/image.jpg"
  }

# Processing nodes
sharpen: lib.pillow.enhance.Sharpen
  image = @image_input.output

contrast: lib.pillow.enhance.AutoContrast
  image = @sharpen.output
  cutoff = 108

# Output node  
output: nodetool.output.Output
  name = "enhanced"
  value = @contrast.output
```

## Grammar (EBNF)

```ebnf
workflow     = { metadata | node_def | edge_def | comment } ;
metadata     = '@' identifier string_value ;
node_def     = identifier ':' node_type [ properties ] ;
properties   = { newline indent property } ;
property     = identifier '=' value ;
edge_def     = node_ref '->' node_ref ;
node_ref     = identifier '.' identifier ;
value        = string | number | boolean | reference | object | list ;
reference    = '@' node_ref ;
object       = '{' [ object_pairs ] '}' ;
object_pairs = object_pair { ',' object_pair } ;
object_pair  = identifier ':' value ;
list         = '[' [ list_items ] ']' ;
list_items   = value { ',' value } ;
string       = '"' { character } '"' ;
number       = [ '-' ] digit { digit } [ '.' digit { digit } ] ;
boolean      = 'true' | 'false' ;
comment      = '#' { character } newline | '/*' { character } '*/' ;
identifier   = letter { letter | digit | '_' } ;
node_type    = identifier { '.' identifier } ;
```

## File Extension

NTL files use the `.ntl` extension.

## API Usage

```python
from nodetool.ntl import load_workflow_from_ntl, parse_ntl

# Load workflow from file
workflow = load_workflow_from_ntl("my_workflow.ntl")

# Parse NTL string
ntl_source = '''
@name "Test"
node1: nodetool.input.StringInput
  value = "hello"
'''
ast = parse_ntl(ntl_source)
```

## Comparison with JSON

NTL is significantly more compact than JSON workflow format:

**JSON** (~40 lines):
```json
{
  "name": "Image Enhance",
  "graph": {
    "nodes": [
      {"id": "1", "type": "nodetool.input.ImageInput", "data": {"name": "image"}},
      {"id": "2", "type": "lib.pillow.enhance.Sharpen", "data": {}},
      {"id": "3", "type": "nodetool.output.Output", "data": {"name": "enhanced"}}
    ],
    "edges": [
      {"source": "1", "sourceHandle": "output", "target": "2", "targetHandle": "image"},
      {"source": "2", "sourceHandle": "output", "target": "3", "targetHandle": "value"}
    ]
  }
}
```

**NTL** (~12 lines):
```ntl
@name "Image Enhance"

input: nodetool.input.ImageInput
  name = "image"

sharpen: lib.pillow.enhance.Sharpen
  image = @input.output

output: nodetool.output.Output
  name = "enhanced"
  value = @sharpen.output
```

~70% reduction in tokens while maintaining full expressiveness.
