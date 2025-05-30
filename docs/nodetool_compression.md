---
layout: default
title: nodetool.compression
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.compression

Nodes for compressing and decompressing data with gzip.

## GzipCompress

Compress bytes using gzip.

Use cases:
- Reduce size of binary data
- Store assets in compressed form
- Prepare data for network transfer

**Tags:** gzip, compress, bytes

**Fields:**
- **data**: Data to compress (bytes)

## GzipDecompress

Decompress gzip data.

Use cases:
- Restore compressed files
- Read data from gzip archives
- Process network payloads

**Tags:** gzip, decompress, bytes

**Fields:**
- **data**: Gzip data to decompress (bytes)
