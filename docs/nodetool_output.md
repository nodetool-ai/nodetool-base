---
layout: default
title: nodetool.output
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.output

Nodes for returning results to the user such as text, images or audio files.

## ArrayOutput

Output node for generic array data.

Use cases:
- Outputting results from machine learning models
- Representing complex numerical data structures

**Tags:** array, numerical

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (NPArray)


## AudioOutput

Output node for audio content references.

Use cases:
- Displaying processed or generated audio
- Passing audio data between workflow nodes
- Returning results of audio analysis

**Tags:** audio, sound, media

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (AudioRef)


## BooleanOutput

Output node for a single boolean value.

Use cases:
- Returning binary results (yes/no, true/false)
- Controlling conditional logic in workflows
- Indicating success/failure of operations

**Tags:** boolean, true, false, flag, condition, flow-control, branch, else, true, false, switch, toggle

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (bool)


## DataframeOutput

Output node for structured data references.

Use cases:
- Outputting tabular data results
- Passing structured data between analysis steps
- Displaying data in table format

**Tags:** dataframe, table, structured

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (DataframeRef)


## DictionaryOutput

Output node for key-value pair data.

Use cases:
- Returning multiple named values
- Passing complex data structures between nodes
- Organizing heterogeneous output data

**Tags:** dictionary, key-value, mapping

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (dict[str, typing.Any])


## DocumentOutput

Output node for document content references.

Use cases:
- Displaying processed or generated documents
- Passing document data between workflow nodes
- Returning results of document analysis

**Tags:** document, pdf, file

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (DocumentRef)


## FloatOutput

Output node for a single float value.

Use cases:
- Returning decimal results (e.g. percentages, ratios)
- Passing floating-point parameters between nodes
- Displaying numeric metrics with decimal precision

**Tags:** float, decimal, number

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (float)


## GroupOutput

Generic output node for grouped data from any node.

Use cases:
- Aggregating multiple outputs from a single node
- Passing varied data types as a single unit
- Organizing related outputs in workflows

**Tags:** group, composite, multi-output

**Fields:**
- **input** (Any)


## ImageListOutput

Output node for a list of image references.

Use cases:
- Displaying multiple images in a grid
- Returning image search results

**Tags:** images, list, gallery

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value**: The images to display. (list[nodetool.metadata.types.ImageRef])


## ImageOutput

Output node for a single image reference.

Use cases:
- Displaying a single processed or generated image
- Passing image data between workflow nodes
- Returning image analysis results

**Tags:** image, picture, visual

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (ImageRef)


## IntegerOutput

Output node for a single integer value.

Use cases:
- Returning numeric results (e.g. counts, indices)
- Passing integer parameters between nodes
- Displaying numeric metrics

**Tags:** integer, number, count

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (int)


## ListOutput

Output node for a list of arbitrary values.

Use cases:
- Returning multiple results from a workflow
- Aggregating outputs from multiple nodes

**Tags:** list, output, any

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (list[typing.Any])


## ModelOutput

Output node for machine learning model references.

Use cases:
- Passing trained models between workflow steps
- Outputting newly created or fine-tuned models
- Referencing models for later use in the workflow

**Tags:** model, ml, ai

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (ModelRef)


## StringOutput

Output node for a single string value.

Use cases:
- Returning text results or messages
- Passing string parameters between nodes
- Displaying short text outputs

**Tags:** string, text, output

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (str)


## TextOutput

Output node for structured text content.

Use cases:
- Returning longer text content or documents
- Passing formatted text between processing steps
- Displaying rich text output

**Tags:** text, content, document

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (TextRef)


## VideoOutput

Output node for video content references.

Use cases:
- Displaying processed or generated video content
- Passing video data between workflow steps
- Returning results of video analysis

**Tags:** video, media, clip

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this output node. (str)
- **value** (VideoRef)


