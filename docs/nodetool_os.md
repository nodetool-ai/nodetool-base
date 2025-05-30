---
layout: default
title: nodetool.os
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.os

File system utilities for working with paths, files and directories.

## AbsolutePath

Return the absolute path of a file or directory.

Use cases:
- Convert relative paths to absolute
- Get full system path
- Resolve path references

**Tags:** files, path, absolute

**Fields:**
- **path**: Path to convert to absolute (str)


## AccessedTime

Get file last accessed timestamp.

**Tags:** files, metadata, accessed, time

**Fields:**
- **path**: Path to file (FilePath)


## Basename

Get the base name component of a file path.

Use cases:
- Extract filename from full path
- Get file name without directory
- Process file names independently

**Tags:** files, path, basename

**Fields:**
- **path**: File path to get basename from (str)
- **remove_extension**: Remove file extension from basename (bool)


## CopyFile

Copy a file from source to destination path.

Use cases:
- Create file backups
- Duplicate files for processing
- Copy files to new locations

**Tags:** files, copy, manage

**Fields:**
- **source_path**: Source file path (FilePath)
- **destination_path**: Destination file path (FilePath)


## CreateDirectory

Create a new directory at specified path.

Use cases:
- Set up directory structure for file organization
- Create output directories for processed files

**Tags:** files, directory, create

**Fields:**
- **path**: Directory path to create (FilePath)
- **exist_ok**: Don't error if directory already exists (bool)


## CreateTemporaryDirectory

Create a temporary directory on disk.

Use cases:
- Provide working directories for intermediate results
- Store ephemeral data across nodes

**Tags:** files, directory, temporary, create

**Fields:**
- **prefix**: Prefix for the directory name (str)
- **suffix**: Suffix for the directory name (str)
- **dir**: Parent directory for the temp directory (str)

## CreateTemporaryFile

Create a temporary file on disk.

Use cases:
- Provide scratch storage for workflows
- Generate unique filenames for intermediate data

**Tags:** files, temporary, create

**Fields:**
- **prefix**: Prefix for the temp file name (str)
- **suffix**: Suffix for the temp file name (str)
- **dir**: Directory where the file is created (str)

## CreatedTime

Get file creation timestamp.

**Tags:** files, metadata, created, time

**Fields:**
- **path**: Path to file (FilePath)


## Dirname

Get the directory name component of a file path.

Use cases:
- Extract directory path from full path
- Get parent directory
- Process directory paths

**Tags:** files, path, dirname

**Fields:**
- **path**: File path to get dirname from (str)


## FileExists

Check if a file or directory exists at the specified path.

Use cases:
- Validate file presence before processing
- Implement conditional logic based on file existence

**Tags:** files, check, exists

**Fields:**
- **path**: Path to check for existence (FilePath)


## FileExtension

Get file extension.

**Tags:** files, metadata, extension

**Fields:**
- **path**: Path to file (FilePath)


## FileName

Get file name without path.

**Tags:** files, metadata, name

**Fields:**
- **path**: Path to file (FilePath)


## FileNameMatch

Match a filename against a pattern using Unix shell-style wildcards.

Use cases:
- Filter files by name pattern
- Validate file naming conventions
- Match file extensions

**Tags:** files, pattern, match, filter

**Fields:**
- **filename**: Filename to check (str)
- **pattern**: Pattern to match against (e.g. *.txt, data_*.csv) (str)
- **case_sensitive**: Whether the pattern matching should be case-sensitive (bool)


## FilterFileNames

Filter a list of filenames using Unix shell-style wildcards.

Use cases:
- Filter multiple files by pattern
- Batch process files matching criteria
- Select files by extension

**Tags:** files, pattern, filter, list

**Fields:**
- **filenames**: list of filenames to filter (list[str])
- **pattern**: Pattern to filter by (e.g. *.txt, data_*.csv) (str)
- **case_sensitive**: Whether the pattern matching should be case-sensitive (bool)


## GetDirectory

Get directory containing the file.

**Tags:** files, metadata, directory

**Fields:**
- **path**: Path to file (FilePath)


## GetEnvironmentVariable

Gets an environment variable value.

Use cases:
- Access configuration
- Get system settings

**Tags:** environment, variable, system

**Fields:**
- **name**: Environment variable name (str)
- **default**: Default value if not found (str | None)


## GetFileSize

Get file size in bytes.

**Tags:** files, metadata, size

**Fields:**
- **path**: Path to file (FilePath)


## GetPathInfo

Gets information about a path.

Use cases:
- Extract path components
- Parse file paths

**Tags:** path, info, metadata

**Fields:**
- **path**: Path to analyze (str)


## GetSystemInfo

Gets system information.

Use cases:
- Check system compatibility
- Platform-specific logic

**Tags:** system, info, platform

**Fields:**


## IsDirectory

Check if path is a directory.

**Tags:** files, metadata, type

**Fields:**
- **path**: Path to check (FilePath)


## IsFile

Check if path is a file.

**Tags:** files, metadata, type

**Fields:**
- **path**: Path to check (FilePath)


## JoinPaths

Joins path components.

Use cases:
- Build file paths
- Create cross-platform paths

**Tags:** path, join, combine

**Fields:**
- **paths**: Path components to join (list[str])


## ListFiles

list files in a directory matching a pattern.

Use cases:
- Get files for batch processing
- Filter files by extension or pattern

**Tags:** files, list, directory

**Fields:**
- **directory**: Directory to scan (FilePath)
- **pattern**: File pattern to match (e.g. *.txt) (str)
- **recursive**: Search subdirectories (bool)


## LoadAudioFile

Read an audio file from disk.

Use cases:
- Load audio for processing
- Import sound files for editing
- Read audio assets for a workflow

**Tags:** audio, input, load, file

**Fields:**
- **path**: Path to the audio file to read (FilePath)


## LoadBytesFile

Read raw bytes from a file on disk.

Use cases:
- Load binary data for processing
- Read binary files for a workflow

**Tags:** files, bytes, read, input, load, file

**Fields:**
- **path**: Path to the file to read (FilePath)


## LoadCSVFile

Read a CSV file from disk.

**Tags:** files, csv, read, input, load, file

**Fields:**
- **path**: Path to the CSV file to read (FilePath)


## LoadDocumentFile

Read a document from disk.

**Tags:** files, document, read, input, load, file

**Fields:**
- **path**: Path to the document to read (FilePath)


## LoadImageFile

Read an image file from disk.

Use cases:
- Load images for processing
- Import photos for editing
- Read image assets for a workflow

**Tags:** image, input, load, file

**Fields:**
- **path**: Path to the image file to read (FilePath)


## LoadVideoFile

Read a video file from disk.

Use cases:
- Load videos for processing
- Import video files for editing
- Read video assets for a workflow

**Tags:** video, input, load, file

**Fields:**
- **path**: Path to the video file to read (str)


## ModifiedTime

Get file last modified timestamp.

**Tags:** files, metadata, modified, time

**Fields:**
- **path**: Path to file (FilePath)


## MoveFile

Move a file from source to destination path.

Use cases:
- Organize files into directories
- Process and archive files
- Relocate completed files

**Tags:** files, move, manage

**Fields:**
- **source_path**: Source file path (FilePath)
- **destination_path**: Destination file path (FilePath)


## NormalizePath

Normalizes a path.

Use cases:
- Standardize paths
- Remove redundant separators

**Tags:** path, normalize, clean

**Fields:**
- **path**: Path to normalize (str)


## PathToString

Convert a FilePath object to a string.

Use cases:
- Get raw string path from FilePath object
- Convert FilePath for string operations
- Extract path string for external use

**Tags:** files, path, string, convert

**Fields:**
- **file_path**: FilePath object to convert to string (FilePath)


## RelativePath

Return a relative path to a target from a start directory.

Use cases:
- Create relative path references
- Generate portable paths
- Compare file locations

**Tags:** files, path, relative

**Fields:**
- **target_path**: Target path to convert to relative (str)
- **start_path**: Start path for relative conversion (str)


## SaveAudioFile

Write an audio file to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** audio, output, save, file

**Fields:**
- **audio**: The audio to save (AudioRef)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: 
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SaveBytesFile

Write raw bytes to a file on disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, bytes, save, output

**Fields:**
- **data**: The bytes to write to file (bytes)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: Name of the file to save. Supports strftime format codes. (str)


## SaveCSVDataframeFile

Write a pandas DataFrame to a CSV file.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, csv, write, output, save, file

**Fields:**
- **dataframe**: DataFrame to write to CSV (DataframeRef)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: Name of the CSV file to save. Supports strftime format codes. (str)


## SaveCSVFile

Write a list of dictionaries to a CSV file.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, csv, write, output, save, file

**Fields:**
- **data**: list of dictionaries to write to CSV (list[dict])
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: Name of the CSV file to save. Supports strftime format codes. (str)


## SaveDocumentFile

Write a document to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** files, document, write, output, save, file

**Fields:**
- **document**: The document to save (DocumentRef)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: Name of the file to save. Supports strftime format codes. (str)


## SaveImageFile

Write an image to disk.

Use cases:
- Save processed images
- Export edited photos
- Archive image results

**Tags:** image, output, save, file

**Fields:**
- **image**: The image to save (ImageRef)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: 
        The name of the image file.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SaveVideoFile

Write a video file to disk.

The filename can include time and date variables:
%Y - Year, %m - Month, %d - Day
%H - Hour, %M - Minute, %S - Second

**Tags:** video, output, save, file

**Fields:**
- **video**: The video to save (VideoRef)
- **folder**: Folder where the file will be saved (FolderPath)
- **filename**: 
        Name of the file to save.
        You can use time and date variables to create unique names:
        %Y - Year
        %m - Month
        %d - Day
        %H - Hour
        %M - Minute
        %S - Second
         (str)


## SetEnvironmentVariable

Sets an environment variable.

Use cases:
- Configure runtime settings
- Set up process environment

**Tags:** environment, variable, system

**Fields:**
- **name**: Environment variable name (str)
- **value**: Environment variable value (str)


## ShowNotification

Shows a system notification.

Use cases:
- Alert user of completed tasks
- Show process status
- Display important messages

**Tags:** notification, system, alert

**Fields:**
- **title**: Title of the notification (str)
- **message**: Content of the notification (str)
- **timeout**: How long the notification should stay visible (in seconds) (int)


## SplitExtension

Split a path into root and extension components.

Use cases:
- Extract file extension
- Process filename without extension
- Handle file types

**Tags:** files, path, extension, split

**Fields:**
- **path**: Path to split (str)


## SplitPath

Split a path into directory and file components.

Use cases:
- Separate directory from filename
- Process path components separately
- Extract path parts

**Tags:** files, path, split

**Fields:**
- **path**: Path to split (str)


