---
layout: default
title: nodetool.data
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.data

Data processing helpers for working with CSV and pandas dataframes.

## CSVRowIterator

Iterate over rows of a CSV string with streaming output.

**Tags:** csv, iterator, stream

**Fields:**
- **csv_data**: CSV formatted text to iterate over (str)
- **delimiter**: Delimiter used in the CSV data (str)

## LoadCSVFileStream

Stream rows from a CSV file on disk one by one.

**Tags:** csv, read, iterator, file, stream

**Fields:**
- **path**: Path to the CSV file to read (FilePath)
- **delimiter**: Delimiter used in the CSV file (str)
