---
layout: default
title: nodetool.audio
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.audio

Utilities for handling audio data. Provides a node for saving audio files to a specified folder.

## SaveAudio

Save an audio file to a specified folder.

Use cases:
- Save generated audio files with timestamps
- Organize outputs into specific folders
- Create backups of generated audio

**Tags:** audio, folder, name

**Fields:**
- **audio** (AudioRef)
- **folder**: The folder to save the audio file to.  (FolderRef)
- **name**: 
        The name of the audio file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)

### required_inputs

**Args:**


