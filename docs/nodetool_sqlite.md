---
layout: default
title: lib.sqlite
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.lib.sqlite

Utilities for interacting with SQLite databases.

## SQLiteQuery

Execute a SQL query on a SQLite database and return the results as a dataframe.

Use cases:
- Run analytics on a local SQLite database
- Load query results into a dataframe
- Combine with other data processing nodes

**Tags:** sqlite, sql, query, database

**Fields:**
- **db_path** (FilePath)
- **query** (str)
- **params** (list[Any])

## SQLiteExecute

Execute a SQL statement on a SQLite database without returning rows.

Use cases:
- Create or modify tables
- Insert or update records
- Run maintenance commands

**Tags:** sqlite, sql, execute, database

**Fields:**
- **db_path** (FilePath)
- **statement** (str)
- **params** (list[Any])
