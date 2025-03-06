---
layout: default
title: nodetool-base Documentation
---

# nodetool-base Documentation
# nodetool.nodes.nodetool.list

## Append

Adds a value to the end of a list.

Use cases:
- Grow a list dynamically
- Add new elements to an existing list
- Implement a stack-like structure

**Tags:** list, add, insert, extend

**Fields:**
- **values** (list[typing.Any])
- **value** (Any)


## Average

Calculates the arithmetic mean of a list of numbers.

Use cases:
- Find average value
- Calculate mean of numeric data

**Tags:** list, average, mean, aggregate, math

**Fields:**
- **values** (list[float])


## Chunk

Splits a list into smaller chunks of specified size.

Use cases:
- Batch processing
- Pagination
- Creating sublists of fixed size

**Tags:** list, chunk, split, group

**Fields:**
- **values** (list[typing.Any])
- **chunk_size** (int)


## Dedupe

Removes duplicate elements from a list, ensuring uniqueness.

Use cases:
- Remove redundant entries
- Create a set-like structure
- Ensure list elements are unique

**Tags:** list, unique, distinct, deduplicate

**Fields:**
- **values** (list[typing.Any])


## Difference

Finds elements that exist in first list but not in second list.

Use cases:
- Find unique elements in one list
- Remove items present in another list
- Identify distinct elements

**Tags:** list, set, difference, subtract

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


## Extend

Merges one list into another, extending the original list.

Use cases:
- Combine multiple lists
- Add all elements from one list to another

**Tags:** list, merge, concatenate, combine

**Fields:**
- **values** (list[typing.Any])
- **other_values** (list[typing.Any])


## FilterDicts

Filter a list of dictionaries based on a condition.

Basic Operators:
- Comparison: >, <, >=, <=, ==, !=
- Logical: and, or, not
- Membership: in, not in

Example Conditions:
# Basic comparisons
age > 30
price <= 100
status == 'active'

# Multiple conditions
age > 30 and salary < 50000
(price >= 100) and (price <= 200)
department in ['Sales', 'Marketing']

# String operations
name.str.startswith('J')
email.str.contains('@company.com')

# Datetime conditions
date > '2024-01-01'
date.dt.year == 2024
date.dt.month >= 6
date.dt.day_name() == 'Monday'

# Date ranges
date.between('2024-01-01', '2024-12-31')
date >= '2024-01-01' and date < '2025-01-01'

# Complex datetime
date.dt.hour < 12
date.dt.dayofweek <= 4  # Weekdays only

# Numeric operations
price.between(100, 200)
quantity % 2 == 0  # Even numbers

# Special values
value.isna()  # Check for NULL/NaN
value.notna()  # Check for non-NULL/non-NaN

Note: Dates should be in ISO format (YYYY-MM-DD) or include time (YYYY-MM-DD HH:MM:SS)

Use cases:
- Filter list of dictionary objects based on criteria
- Extract subset of data meeting specific conditions
- Clean data by removing unwanted entries

**Tags:** list, filter, query, condition

**Fields:**
- **values** (list[dict])
- **condition**: 
        The filtering condition using pandas query syntax.

        Basic Operators:
        - Comparison: >, <, >=, <=, ==, !=
        - Logical: and, or, not
        - Membership: in, not in
        
        Example Conditions:
        # Basic comparisons
        age > 30
        price <= 100
        status == 'active'
        
        See node documentation for more examples.
         (str)


## FilterDictsByNumber

Filters a list of dictionaries based on numeric values for a specified key.

Use cases:
- Filter dictionaries by numeric comparisons (greater than, less than, equal to)
- Filter records with even/odd numeric values
- Filter entries with positive/negative numbers

**Tags:** list, filter, dictionary, numbers, numeric

**Fields:**
- **values** (list[dict])
- **key** (str)
- **filter_type** (FilterDictNumberType)
- **value** (float | None)


## FilterDictsByRange

Filters a list of dictionaries based on a numeric range for a specified key.

Use cases:
- Filter records based on numeric ranges (e.g., price range, age range)
- Find entries with values within specified bounds
- Filter data sets based on numeric criteria

**Tags:** list, filter, dictionary, range, between

**Fields:**
- **values** (list[dict])
- **key**: The dictionary key to check for the range (str)
- **min_value**: The minimum value (inclusive) of the range (float)
- **max_value**: The maximum value (inclusive) of the range (float)
- **inclusive**: If True, includes the min and max values in the results (bool)


## FilterDictsByValue

Filters a list of dictionaries based on their values using various criteria.

Use cases:
- Filter dictionaries by value content
- Filter dictionaries by value type
- Filter dictionaries by value patterns

**Tags:** list, filter, dictionary, values

**Fields:**
- **values** (list[dict])
- **key**: The dictionary key to check (str)
- **filter_type**: The type of filter to apply (FilterType)
- **criteria**: The filtering criteria (text to match, type name, or length as string) (str)


## FilterDictsRegex

Filters a list of dictionaries using regular expressions on specified keys.

Use cases:
- Filter dictionaries with values matching complex patterns
- Search for dictionaries containing emails, dates, or specific formats
- Advanced text pattern matching across dictionary values

**Tags:** list, filter, regex, dictionary, pattern

**Fields:**
- **values** (list[dict])
- **key** (str)
- **pattern** (str)
- **full_match** (bool)


## FilterNone

Filters out None values from a list.

Use cases:
- Clean data by removing null values
- Get only valid entries
- Remove placeholder values

**Tags:** list, filter, none, null

**Fields:**
- **values** (list[typing.Any])


## FilterNumberRange

Filters a list of numbers to find values within a specified range.

Use cases:
- Find numbers within a specific range
- Filter data points within bounds
- Implement range-based filtering

**Tags:** list, filter, numbers, range, between

**Fields:**
- **values** (list[float])
- **min_value** (float)
- **max_value** (float)
- **inclusive** (bool)


## FilterNumbers

Filters a list of numbers based on various numerical conditions.

Use cases:
- Filter numbers by comparison (greater than, less than, equal to)
- Filter even/odd numbers
- Filter positive/negative numbers

**Tags:** list, filter, numbers, numeric

**Fields:**
- **values** (list[float])
- **filter_type**: The type of filter to apply (FilterNumberType)
- **value**: The comparison value (for greater_than, less_than, equal_to) (float | None)


## FilterRegex

Filters a list of strings using regular expressions.

Use cases:
- Filter strings using complex patterns
- Extract strings matching specific formats (emails, dates, etc.)
- Advanced text pattern matching

**Tags:** list, filter, regex, pattern, text

**Fields:**
- **values** (list[str])
- **pattern**: The regular expression pattern to match against. (str)
- **full_match**: Whether to match the entire string or find pattern anywhere in string (bool)


## FilterStrings

Filters a list of strings based on various criteria.

Use cases:
- Filter strings by length
- Filter strings containing specific text
- Filter strings by prefix/suffix
- Filter strings using regex patterns

**Tags:** list, filter, strings, text

**Fields:**
- **values** (list[str])
- **filter_type**: The type of filter to apply (FilterType)
- **criteria**: The filtering criteria (text to match or length as string) (str)


## Flatten

Flattens a nested list structure into a single flat list.

Use cases:
- Convert nested lists into a single flat list
- Simplify complex list structures
- Process hierarchical data as a sequence

Examples:
[[1, 2], [3, 4]] -> [1, 2, 3, 4]
[[1, [2, 3]], [4, [5, 6]]] -> [1, 2, 3, 4, 5, 6]

**Tags:** list, flatten, nested, structure

**Fields:**
- **values** (list[typing.Any])
- **max_depth** (int)


## GenerateSequence

Generates a list of integers within a specified range.

Use cases:
- Create numbered lists
- Generate index sequences
- Produce arithmetic progressions

**Tags:** list, range, sequence, numbers

**Fields:**
- **start** (int)
- **stop** (int)
- **step** (int)


## GetElement

Retrieves a single value from a list at a specific index.

Use cases:
- Access a specific element by position
- Implement array-like indexing
- Extract the first or last element

**Tags:** list, get, extract, value

**Fields:**
- **values** (list[typing.Any])
- **index** (int)


## Intersection

Finds common elements between two lists.

Use cases:
- Find elements present in both lists
- Identify shared items between collections
- Filter for matching elements

**Tags:** list, set, intersection, common

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


## Length

Calculates the length of a list.

Use cases:
- Determine the number of elements in a list
- Check if a list is empty
- Validate list size constraints

**Tags:** list, count, size

**Fields:**
- **values** (list[typing.Any])


## MapField

Extracts a specific field from a list of dictionaries or objects.

Use cases:
- Extract specific fields from a list of objects
- Transform complex data structures into simple lists
- Collect values for a particular key across multiple dictionaries

**Tags:** list, map, field, extract, pluck

**Fields:**
- **values** (list[dict | object])
- **field** (str)
- **default** (Any)


## MapTemplate

Maps a template string over a list of dictionaries or objects using Jinja2 templating.

Use cases:
- Formatting multiple records into strings
- Generating text from structured data
- Creating text representations of data collections

Examples:
- template: "Name: {{ name }}, Age: {{ age }}"
values: [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
-> ["Name: Alice, Age: 30", "Name: Bob, Age: 25"]

Available filters:
- truncate(length): Truncates text to given length
- upper: Converts text to uppercase
- lower: Converts text to lowercase
- title: Converts text to title case
- trim: Removes whitespace from start/end
- replace(old, new): Replaces substring
- default(value): Sets default if value is undefined
- first: Gets first character/item
- last: Gets last character/item
- length: Gets length of string/list
- sort: Sorts list
- join(delimiter): Joins list with delimiter

**Tags:** list, template, map, formatting

**Fields:**
- **template**: 
        Template string with Jinja2 placeholders for formatting
        Examples:
        - "Name: {{ name }}, Age: {{ age }}"
        - "{{ title|truncate(20) }}"
        - "{{ name|upper }}"
         (str)
- **values** (list[dict[str, typing.Any] | object])


## Maximum

Finds the largest value in a list of numbers.

Use cases:
- Find highest value
- Get largest number in dataset

**Tags:** list, max, maximum, aggregate, math

**Fields:**
- **values** (list[float])


## Minimum

Finds the smallest value in a list of numbers.

Use cases:
- Find lowest value
- Get smallest number in dataset

**Tags:** list, min, minimum, aggregate, math

**Fields:**
- **values** (list[float])


## Product

Calculates the product of all numbers in a list.

Use cases:
- Multiply all numbers together
- Calculate compound values

**Tags:** list, product, multiply, aggregate, math

**Fields:**
- **values** (list[float])


## Randomize

Randomly shuffles the elements of a list.

Use cases:
- Randomize the order of items in a playlist
- Implement random sampling without replacement
- Create randomized data sets for testing

**Tags:** list, shuffle, random, order

**Fields:**
- **values** (list[typing.Any])


## Reverse

Inverts the order of elements in a list.

Use cases:
- Reverse the order of a sequence

**Tags:** list, reverse, invert, flip

**Fields:**
- **values** (list[typing.Any])


## SaveList

Saves a list to a text file, placing each element on a new line.

Use cases:
- Export list data to a file
- Create a simple text-based database
- Generate line-separated output

**Tags:** list, save, file, serialize

**Fields:**
- **values** (list[typing.Any])
- **name**: 
        Name of the output file.
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


## SelectElements

Selects specific values from a list using index positions.

Use cases:
- Pick specific elements by their positions
- Rearrange list elements
- Create a new list from selected indices

**Tags:** list, select, index, extract

**Fields:**
- **values** (list[typing.Any])
- **indices** (list[int])


## Slice

Extracts a subset from a list using start, stop, and step indices.

Use cases:
- Get a portion of a list
- Implement pagination
- Extract every nth element

**Tags:** list, slice, subset, extract

**Fields:**
- **values** (list[typing.Any])
- **start** (int)
- **stop** (int)
- **step** (int)


## Sort

Sorts the elements of a list in ascending or descending order.

Use cases:
- Organize data in a specific order
- Prepare data for binary search or other algorithms
- Rank items based on their values

**Tags:** list, sort, order, arrange

**Fields:**
- **values** (list[typing.Any])
- **order** (SortOrder)


## Sum

Calculates the sum of a list of numbers.

Use cases:
- Calculate total of numeric values
- Add up all elements in a list

**Tags:** list, sum, aggregate, math

**Fields:**
- **values** (list[float])


## Transform

Applies a transformation to each element in a list.

Use cases:
- Convert types (str to int, etc.)
- Apply formatting
- Mathematical operations

**Tags:** list, transform, map, convert

**Fields:**
- **values** (list[typing.Any])
- **transform_type** (TransformType)


## Union

Combines unique elements from two lists.

Use cases:
- Merge lists while removing duplicates
- Combine collections uniquely
- Create comprehensive set of items

**Tags:** list, set, union, combine

**Fields:**
- **list1** (list[typing.Any])
- **list2** (list[typing.Any])


# nodetool.nodes.nodetool.control

## If

Conditionally executes one of two branches based on a condition.

Use cases:
- Branch workflow based on conditions
- Handle different cases in data processing
- Implement decision logic

**Tags:** control, flow, condition, logic, else, true, false, switch, toggle, flow-control

**Fields:**
- **condition**: The condition to evaluate (bool)
- **value**: The value to pass to the next node (Any)


# nodetool.nodes.nodetool.code

## EvaluateExpression

Evaluates a Python expression with safety restrictions.

Use cases:
- Calculate values dynamically
- Transform data with simple expressions
- Quick data validation

IMPORTANT: Only enabled in non-production environments

**Tags:** python, expression, evaluate

**Fields:**
- **expression**: Python expression to evaluate. Variables are available as locals. (str)
- **variables**: Variables available to the expression (dict[str, typing.Any])


## ExecutePython

Executes Python code with safety restrictions.

Use cases:
- Run custom data transformations
- Prototype node functionality
- Debug and testing workflows

IMPORTANT: Only enabled in non-production environments

**Tags:** python, code, execute

**Fields:**
- **code**: Python code to execute. Input variables are available as locals. Assign the desired output to the 'result' variable. (str)
- **inputs**: Input variables available to the code as locals. (dict[str, typing.Any])


# nodetool.nodes.nodetool.os

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


# nodetool.nodes.nodetool.boolean

## All

Checks if all boolean values in a list are True.


Use cases:
- Ensure all conditions in a set are met
- Implement comprehensive checks
- Validate multiple criteria simultaneously

**Tags:** boolean, all, check, logic, condition, flow-control, branch

**Fields:**
- **values**: List of boolean values to check (list[bool])


## Compare

Compares two values using a specified comparison operator.

Use cases:
- Implement decision points in workflows
- Filter data based on specific criteria
- Create dynamic thresholds or limits

**Tags:** compare, condition, logic

**Fields:**
- **a**: First value to compare (Any)
- **b**: Second value to compare (Any)
- **comparison**: Comparison operator to use (Comparison)


## ConditionalSwitch

Performs a conditional check on a boolean input and returns a value based on the result.

Use cases:
- Implement conditional logic in workflows
- Create dynamic branches in workflows
- Implement decision points in workflows

**Tags:** if, condition, flow-control, branch, true, false, switch, toggle

**Fields:**
- **condition**: The condition to check (bool)
- **if_true**: The value to return if the condition is true (Any)
- **if_false**: The value to return if the condition is false (Any)


## IsIn

Checks if a value is present in a list of options.

Use cases:
- Validate input against a set of allowed values
- Implement category or group checks
- Filter data based on inclusion criteria

**Tags:** membership, contains, check

**Fields:**
- **value**: The value to check for membership (Any)
- **options**: The list of options to check against (list[typing.Any])


## IsNone

Checks if a value is None.

Use cases:
- Validate input presence
- Handle optional parameters
- Implement null checks in data processing

**Tags:** null, none, check

**Fields:**
- **value**: The value to check for None (Any)


## LogicalOperator

Performs logical operations on two boolean inputs.

Use cases:
- Combine multiple conditions in decision-making
- Implement complex logical rules in workflows
- Create advanced filters or triggers

**Tags:** boolean, logic, operator, condition, flow-control, branch, else, true, false, switch, toggle

**Fields:**
- **a**: First boolean input (bool)
- **b**: Second boolean input (bool)
- **operation**: Logical operation to perform (BooleanOperation)


## Not

Performs logical NOT operation on a boolean input.

Use cases:
- Invert a condition's result
- Implement toggle functionality
- Create opposite logic branches

**Tags:** boolean, logic, not, invert, !, negation, condition, else, true, false, switch, toggle, flow-control, branch

**Fields:**
- **value**: Boolean input to negate (bool)


## Some

Checks if any boolean value in a list is True.

Use cases:
- Check if at least one condition in a set is met
- Implement optional criteria checks
- Create flexible validation rules

**Tags:** boolean, any, check, logic, condition, flow-control, branch

**Fields:**
- **values**: List of boolean values to check (list[bool])


# nodetool.nodes.nodetool.input

## AudioInput

Audio asset input for workflows.

Use cases:
- Load audio files for processing
- Analyze sound or speech content
- Provide audio input to models

**Tags:** input, parameter, audio

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The audio to use as input. (AudioRef)


## BooleanInput

Boolean parameter input for workflows.

Use cases:
- Toggle features on/off
- Set binary flags
- Control conditional logic

**Tags:** input, parameter, boolean, bool

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value** (bool)


## ChatInput

Chat message input for workflows.

Use cases:
- Accept user prompts or queries
- Capture conversational input
- Provide instructions to language models

**Tags:** input, parameter, chat, message

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The chat message to use as input. (list[nodetool.metadata.types.Message])


## CollectionInput

Collection input for workflows.

Use cases:
- Select a vector database collection
- Specify target collection for indexing
- Choose collection for similarity search

**Tags:** input, parameter, collection, chroma

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The collection to use as input. (Collection)


## DocumentFileInput

Document file input for workflows.

Use cases:
- Load text documents for processing
- Analyze document content
- Extract text for NLP tasks
- Index documents for search

**Tags:** input, parameter, document, text

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The path to the document file. (FilePath)


## DocumentInput

Document asset input for workflows.

Use cases:
- Load documents for processing
- Analyze document content
- Provide document input to models

**Tags:** input, parameter, document

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The document to use as input. (DocumentRef)


## EnumInput

Enumeration parameter input for workflows.

Use cases:
- Select from predefined options
- Enforce choice from valid values
- Configure categorical parameters

**Tags:** input, parameter, enum, options, select

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value** (str)
- **options**: Comma-separated list of valid options (str)


## FloatInput

Float parameter input for workflows.

Use cases:
- Specify a numeric value within a defined range
- Set thresholds or scaling factors
- Configure continuous parameters like opacity or volume

**Tags:** input, parameter, float, number

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value** (float)
- **min** (float)
- **max** (float)


## GroupInput

Generic group input for loops.

Use cases:
- provides input for a loop
- iterates over a group of items

**Tags:** input, group, collection, loop

**Fields:**


## ImageInput

Image asset input for workflows.

Use cases:
- Load images for processing or analysis
- Provide visual input to models
- Select images for manipulation

**Tags:** input, parameter, image

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The image to use as input. (ImageRef)


## IntegerInput

Integer parameter input for workflows.

Use cases:
- Specify counts or quantities
- Set index values
- Configure discrete numeric parameters

**Tags:** input, parameter, integer, number

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value** (int)
- **min** (int)
- **max** (int)


## PathInput

Local path input for workflows.

Use cases:
- Provide a local path to a file or directory
- Specify a file or directory for processing
- Load local data for analysis

**Tags:** input, parameter, path

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The path to use as input. (FilePath)


## StringInput

String parameter input for workflows.

Use cases:
- Provide text labels or names
- Enter search queries
- Specify file paths or URLs

**Tags:** input, parameter, string, text

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value** (str)


## TextInput

Text content input for workflows.

Use cases:
- Load text documents or articles
- Process multi-line text content
- Analyze large text bodies

**Tags:** input, parameter, text

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The text to use as input. (TextRef)


## VideoInput

Video asset input for workflows.

Use cases:
- Load video files for processing
- Analyze video content
- Extract frames or audio from videos

**Tags:** input, parameter, video

**Fields:**
- **name**: The parameter name for the workflow. (str)
- **description**: The description for this input node. (str)
- **value**: The video to use as input. (VideoRef)


# nodetool.nodes.nodetool.text

## Chunk

Splits text into chunks of specified word length.

Use cases:
- Preparing text for processing by models with input length limits
- Creating manageable text segments for parallel processing
- Generating summaries of text sections

**Tags:** text, chunk, split

**Fields:**
- **text** (str)
- **length** (int)
- **overlap** (int)
- **separator** (str | None)


## Concat

Concatenates two text inputs into a single output.

Use cases:
- Joining outputs from multiple text processing nodes
- Combining parts of sentences or paragraphs
- Merging text data from different sources

**Tags:** text, concatenation, combine, +

**Fields:**
- **a** (str)
- **b** (str)


## Contains

Checks if text contains a specified substring.

Use cases:
- Searching for keywords in text
- Filtering content based on presence of terms
- Validating text content

**Tags:** text, check, contains, compare, validate, substring, string

**Fields:**
- **text** (str)
- **substring** (str)
- **case_sensitive** (bool)


## CountTokens

Counts the number of tokens in text using tiktoken.

Use cases:
- Checking text length for LLM input limits
- Estimating API costs
- Managing token budgets in text processing

**Tags:** text, tokens, count, encoding

**Fields:**
- **text** (str)
- **encoding**: The tiktoken encoding to use for token counting (TiktokenEncoding)


## EndsWith

Checks if text ends with a specified suffix.

Use cases:
- Validating file extensions
- Checking string endings
- Filtering text based on ending content

**Tags:** text, check, suffix, compare, validate, substring, string

**Fields:**
- **text** (str)
- **suffix** (str)


## Extract

Extracts a substring from input text.

Use cases:
- Extracting specific portions of text for analysis
- Trimming unwanted parts from text data
- Focusing on relevant sections of longer documents

**Tags:** text, extract, substring

**Fields:**
- **text** (str)
- **start** (int)
- **end** (int)


## ExtractJSON

Extracts data from JSON using JSONPath expressions.

Use cases:
- Retrieving specific fields from complex JSON structures
- Filtering and transforming JSON data for analysis
- Extracting nested data from API responses or configurations

**Tags:** json, extract, jsonpath

**Fields:**
- **text** (str)
- **json_path** (str)
- **find_all** (bool)


## ExtractRegex

Extracts substrings matching regex groups from text.

Use cases:
- Extracting structured data (e.g., dates, emails) from unstructured text
- Parsing specific patterns in log files or documents
- Isolating relevant information from complex text formats

**Tags:** text, regex, extract

**Fields:**
- **text** (str)
- **regex** (str)
- **dotall** (bool)
- **ignorecase** (bool)
- **multiline** (bool)


## FindAllRegex

Finds all regex matches in text as separate substrings.

Use cases:
- Identifying all occurrences of a pattern in text
- Extracting multiple instances of structured data
- Analyzing frequency and distribution of specific text patterns

**Tags:** text, regex, find

**Fields:**
- **text** (str)
- **regex** (str)
- **dotall** (bool)
- **ignorecase** (bool)
- **multiline** (bool)


## FormatText

Replaces placeholders in a string with dynamic inputs using Jinja2 templating.

Use cases:
- Generating personalized messages with dynamic content
- Creating parameterized queries or commands
- Formatting and filtering text output based on variable inputs

Examples:
- text: "Hello, {{ name }}!"
- text: "Title: {{ title|truncate(20) }}"
- text: "Name: {{ name|upper }}"

Available filters:
- truncate(length): Truncates text to given length
- upper: Converts text to uppercase
- lower: Converts text to lowercase
- title: Converts text to title case
- trim: Removes whitespace from start/end
- replace(old, new): Replaces substring
- default(value): Sets default if value is undefined
- first: Gets first character/item
- last: Gets last character/item
- length: Gets length of string/list
- sort: Sorts list
- join(delimiter): Joins list with delimiter

**Tags:** text, template, formatting

**Fields:**
- **template**: 
    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}" 

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
 (str)


## HasLength

Checks if text length meets specified conditions.

Use cases:
- Validating input length requirements
- Filtering text by length
- Checking content size constraints

**Tags:** text, check, length, compare, validate, whitespace, string

**Fields:**
- **text** (str)
- **min_length** (int | None)
- **max_length** (int | None)
- **exact_length** (int | None)


## IsEmpty

Checks if text is empty or contains only whitespace.

Use cases:
- Validating required text fields
- Filtering out empty content
- Checking for meaningful input

**Tags:** text, check, empty, compare, validate, whitespace, string

**Fields:**
- **text** (str)
- **trim_whitespace** (bool)


## Join

Joins a list of strings into a single string using a specified separator.

Use cases:
- Combining multiple text elements with a consistent delimiter
- Creating comma-separated lists from individual items
- Assembling formatted text from array elements

**Tags:** text, join, combine, +, add, concatenate

**Fields:**
- **strings** (list[str])
- **separator** (str)


## ParseJSON

Parses a JSON string into a Python object.

Use cases:
- Converting JSON API responses for further processing
- Preparing structured data for analysis or storage
- Extracting configuration or settings from JSON files

**Tags:** json, parse, convert

**Fields:**
- **text** (str)


## RegexMatch

Find all matches of a regex pattern in text.

Use cases:
- Extract specific patterns from text
- Validate text against patterns
- Find all occurrences of a pattern

**Tags:** regex, search, pattern, match

**Fields:**
- **text**: Text to search in (str)
- **pattern**: Regular expression pattern (str)
- **group**: Capture group to extract (0 for full match) (int | None)


## RegexReplace

Replace text matching a regex pattern.

Use cases:
- Clean or standardize text
- Remove unwanted patterns
- Transform text formats

**Tags:** regex, replace, substitute

**Fields:**
- **text**: Text to perform replacements on (str)
- **pattern**: Regular expression pattern (str)
- **replacement**: Replacement text (str)
- **count**: Maximum replacements (0 for unlimited) (int)


## RegexSplit

Split text using a regex pattern as delimiter.

Use cases:
- Parse structured text
- Extract fields from formatted strings
- Tokenize text

**Tags:** regex, split, tokenize

**Fields:**
- **text**: Text to split (str)
- **pattern**: Regular expression pattern to split on (str)
- **maxsplit**: Maximum number of splits (0 for unlimited) (int)


## RegexValidate

Check if text matches a regex pattern.

Use cases:
- Validate input formats (email, phone, etc)
- Check text structure
- Filter text based on patterns

**Tags:** regex, validate, check

**Fields:**
- **text**: Text to validate (str)
- **pattern**: Regular expression pattern (str)


## Replace

Replaces a substring in a text with another substring.

Use cases:
- Correcting or updating specific text patterns
- Sanitizing or normalizing text data
- Implementing simple text transformations

**Tags:** text, replace, substitute

**Fields:**
- **text** (str)
- **old** (str)
- **new** (str)


## SaveText

Saves input text to a file in the assets folder.

Use cases:
- Persisting processed text results
- Creating text files for downstream nodes or external use
- Archiving text data within the workflow

**Tags:** text, save, file

**Fields:**
- **text** (str)
- **folder**: Name of the output folder. (FolderRef)
- **name**: 
        Name of the output file.
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


## Slice

Slices text using Python's slice notation (start:stop:step).

Use cases:
- Extracting specific portions of text with flexible indexing
- Reversing text using negative step
- Taking every nth character with step parameter

Examples:
- start=0, stop=5: first 5 characters
- start=-5: last 5 characters
- step=2: every second character
- step=-1: reverse the text

**Tags:** text, slice, substring

**Fields:**
- **text** (str)
- **start** (int | None)
- **stop** (int | None)
- **step** (int | None)


## Split

Separates text into a list of strings based on a specified delimiter.

Use cases:
- Parsing CSV or similar delimited data
- Breaking down sentences into words or phrases
- Extracting specific elements from structured text

**Tags:** text, split, tokenize

**Fields:**
- **text** (str)
- **delimiter** (str)


## StartsWith

Checks if text starts with a specified prefix.

Use cases:
- Validating string prefixes
- Filtering text based on starting content
- Checking file name patterns

**Tags:** text, check, prefix, compare, validate, substring, string

**Fields:**
- **text** (str)
- **prefix** (str)


## Template

Uses Jinja2 templating to format strings with variables and filters.

Use cases:
- Generating personalized messages with dynamic content
- Creating parameterized queries or commands
- Formatting and filtering text output based on variable inputs

Examples:
- text: "Hello, {{ name }}!"
- text: "Title: {{ title|truncate(20) }}"
- text: "Name: {{ name|upper }}"

Available filters:
- truncate(length): Truncates text to given length
- upper: Converts text to uppercase
- lower: Converts text to lowercase
- title: Converts text to title case
- trim: Removes whitespace from start/end
- replace(old, new): Replaces substring
- default(value): Sets default if value is undefined
- first: Gets first character/item
- last: Gets last character/item
- length: Gets length of string/list
- sort: Sorts list
- join(delimiter): Joins list with delimiter

**Tags:** text, template, formatting, format, combine, concatenate, +, add, variable, replace, filter

**Fields:**
- **string**: 
    Examples:
    - text: "Hello, {{ name }}!"
    - text: "Title: {{ title|truncate(20) }}"
    - text: "Name: {{ name|upper }}"

    Available filters:
    - truncate(length): Truncates text to given length
    - upper: Converts text to uppercase
    - lower: Converts text to lowercase
    - title: Converts text to title case
    - trim: Removes whitespace from start/end
    - replace(old, new): Replaces substring
    - default(value): Sets default if value is undefined
    - first: Gets first character/item
    - last: Gets last character/item
    - length: Gets length of string/list
    - sort: Sorts list
    - join(delimiter): Joins list with delimiter
 (str)
- **values**: 
        The values to replace in the string.
        - If a string, it will be used as the format string.
        - If a list, it will be used as the format arguments.
        - If a dictionary, it will be used as the template variables.
        - If an object, it will be converted to a dictionary using the object's __dict__ method.
         (str | list | dict[str, typing.Any] | object)


# nodetool.nodes.nodetool.group

## Loop

Loops over a list of items and processes the remaining nodes for each item.

Use cases:
- Loop over a list of items and process the nodes inside the group

**Tags:** loop, itereate, repeat, for, each, batch

**Fields:**
- **input**: The input data to loop over. (Any)


# nodetool.nodes.nodetool.audio

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


# nodetool.nodes.nodetool.math

## Add

Performs addition on two inputs.

**Tags:** math, plus, add, addition, sum, +

**Fields:**
- **a** (int | float)
- **b** (int | float)


## BinaryOperation

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Cosine

Computes the cosine of input angles in radians.

Use cases:
- Calculating horizontal components in physics
- Creating circular motions
- Phase calculations in signal processing

**Tags:** math, trigonometry, cosine, cos

**Fields:**
- **angle_rad** (float | int)


## Divide

Divides the first input by the second.

**Tags:** math, division, arithmetic, quotient, /

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Modulus

Calculates the element-wise remainder of division.

Use cases:
- Implementing cyclic behaviors
- Checking for even/odd numbers
- Limiting values to a specific range

**Tags:** math, modulo, remainder, mod, %

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Multiply

Multiplies two inputs.

**Tags:** math, product, times, *

**Fields:**
- **a** (int | float)
- **b** (int | float)


## Power

Raises the base to the power of the exponent element-wise.

Use cases:
- Calculating compound interest
- Implementing polynomial functions
- Applying non-linear transformations to data

**Tags:** math, exponentiation, power, pow, **

**Fields:**
- **base** (float | int)
- **exponent** (float | int)


## Sine

Computes the sine of input angles in radians.

Use cases:
- Calculating vertical components in physics
- Generating smooth periodic functions
- Audio signal processing

**Tags:** math, trigonometry, sine, sin

**Fields:**
- **angle_rad** (float | int)


## Sqrt

Calculates the square root of the input element-wise.

Use cases:
- Normalizing data
- Calculating distances in Euclidean space
- Finding intermediate values in binary search

**Tags:** math, square root, sqrt, √

**Fields:**
- **x** (int | float)


## Subtract

Subtracts the second input from the first.

**Tags:** math, minus, difference, -

**Fields:**
- **a** (int | float)
- **b** (int | float)


# nodetool.nodes.nodetool.constant

## Audio

Represents an audio file constant in the workflow.

Use cases:
- Provide a fixed audio input for audio processing nodes
- Reference a specific audio file in the workflow
- Set default audio for testing or demonstration purposes

**Tags:** audio, file, mp3, wav

**Fields:**
- **value** (AudioRef)


## Bool

Represents a boolean constant in the workflow.

Use cases:
- Control flow decisions in conditional nodes
- Toggle features or behaviors in the workflow
- Set default boolean values for configuration

**Tags:** boolean, logic, flag

**Fields:**
- **value** (bool)


## Constant

**Fields:**


## DataFrame

Represents a fixed DataFrame constant in the workflow.

Use cases:
- Provide static data for analysis or processing
- Define lookup tables or reference data
- Set sample data for testing or demonstration

**Tags:** table, data, dataframe, pandas

**Fields:**
- **value** (DataframeRef)


## Date

Make a date object from year, month, day.

**Tags:** date, make, create

**Fields:**
- **year**: Year of the date (int)
- **month**: Month of the date (int)
- **day**: Day of the date (int)


## DateTime

Make a datetime object from year, month, day, hour, minute, second.

**Tags:** datetime, make, create

**Fields:**
- **year**: Year of the datetime (int)
- **month**: Month of the datetime (int)
- **day**: Day of the datetime (int)
- **hour**: Hour of the datetime (int)
- **minute**: Minute of the datetime (int)
- **second**: Second of the datetime (int)
- **microsecond**: Microsecond of the datetime (int)
- **tzinfo**: Timezone of the datetime (str)
- **utc_offset**: UTC offset of the datetime (int)


## Dict

Represents a dictionary constant in the workflow.

Use cases:
- Store configuration settings
- Provide structured data inputs
- Define parameter sets for other nodes

**Tags:** dictionary, key-value, mapping

**Fields:**
- **value** (dict[str, typing.Any])


## Document

Represents a document constant in the workflow.

**Tags:** document, pdf, word, docx

**Fields:**
- **value** (DocumentRef)


## Float

Represents a floating-point number constant in the workflow.

Use cases:
- Set numerical parameters for calculations
- Define thresholds or limits
- Provide fixed numerical inputs for processing

**Tags:** number, decimal, float

**Fields:**
- **value** (float)


## Image

Represents an image file constant in the workflow.

Use cases:
- Provide a fixed image input for image processing nodes
- Reference a specific image file in the workflow
- Set default image for testing or demonstration purposes

**Tags:** picture, photo, image

**Fields:**
- **value** (ImageRef)


## Integer

Represents an integer constant in the workflow.

Use cases:
- Set numerical parameters for calculations
- Define counts, indices, or sizes
- Provide fixed numerical inputs for processing

**Tags:** number, integer, whole

**Fields:**
- **value** (int)


## JSON

Represents a JSON constant in the workflow.

**Tags:** json, object, dictionary

**Fields:**
- **value** (JSONRef)


## List

Represents a list constant in the workflow.

Use cases:
- Store multiple values of the same type
- Provide ordered data inputs
- Define sequences for iteration in other nodes

**Tags:** array, sequence, collection

**Fields:**
- **value** (list[typing.Any])


## String

Represents a string constant in the workflow.

Use cases:
- Provide fixed text inputs for processing
- Define labels, identifiers, or names
- Set default text values for configuration

**Tags:** text, string, characters

**Fields:**
- **value** (str)


## Video

Represents a video file constant in the workflow.

Use cases:
- Provide a fixed video input for video processing nodes
- Reference a specific video file in the workflow
- Set default video for testing or demonstration purposes

**Tags:** video, movie, mp4, file

**Fields:**
- **value** (VideoRef)


# nodetool.nodes.nodetool.dictionary

## ArgMax

Returns the label associated with the highest value in a dictionary.

Use cases:
- Get the most likely class from classification probabilities
- Find the category with highest score
- Identify the winner in a voting/ranking system

**Tags:** dictionary, maximum, label, argmax

**Fields:**
- **scores**: Dictionary mapping labels to their corresponding scores/values (dict[str, float])


## Combine

Merges two dictionaries, with second dictionary values taking precedence.

Use cases:
- Combine default and custom configurations
- Merge partial updates with existing data
- Create aggregate data structures

**Tags:** dictionary, merge, update, +, add, concatenate

**Fields:**
- **dict_a** (dict[str, typing.Any])
- **dict_b** (dict[str, typing.Any])


## Filter

Creates a new dictionary with only specified keys from the input.

Use cases:
- Extract relevant fields from a larger data structure
- Implement data access controls
- Prepare specific data subsets for processing

**Tags:** dictionary, filter, select

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **keys** (list[str])


## GetValue

Retrieves a value from a dictionary using a specified key.

Use cases:
- Access a specific item in a configuration dictionary
- Retrieve a value from a parsed JSON object
- Extract a particular field from a data structure

**Tags:** dictionary, get, value, key

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **key** (str)
- **default** (Any)


## MakeDictionary

Creates a simple dictionary with up to three key-value pairs.

Use cases:
- Create configuration entries
- Initialize simple data structures
- Build basic key-value mappings

**Tags:** dictionary, create, simple

**Fields:**


## ParseJSON

Parses a JSON string into a Python dictionary.

Use cases:
- Process API responses
- Load configuration files
- Deserialize stored data

**Tags:** json, parse, dictionary

**Fields:**
- **json_string** (str)


## ReduceDictionaries

Reduces a list of dictionaries into one dictionary based on a specified key field.

Use cases:
- Aggregate data by a specific field
- Create summary dictionaries from list of records
- Combine multiple data points into a single structure

**Tags:** dictionary, reduce, aggregate

**Fields:**
- **dictionaries**: List of dictionaries to be reduced (list[dict[str, typing.Any]])
- **key_field**: The field to use as the key in the resulting dictionary (str)
- **value_field**: Optional field to use as the value. If not specified, the entire dictionary (minus the key field) will be used as the value. (str | None)
- **conflict_resolution**: How to handle conflicts when the same key appears multiple times (ConflictResolution)


## Remove

Removes a key-value pair from a dictionary.

Use cases:
- Delete a specific configuration option
- Remove sensitive information before processing
- Clean up temporary entries in a data structure

**Tags:** dictionary, remove, delete

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **key** (str)


## Update

Updates a dictionary with new key-value pairs.

Use cases:
- Extend a configuration with additional settings
- Add new entries to a cache or lookup table
- Merge user input with existing data

**Tags:** dictionary, add, update

**Fields:**
- **dictionary** (dict[str, typing.Any])
- **new_pairs** (dict[str, typing.Any])


## Zip

Creates a dictionary from parallel lists of keys and values.

Use cases:
- Convert separate data columns into key-value pairs
- Create lookups from parallel data structures
- Transform list data into associative arrays

**Tags:** dictionary, create, zip

**Fields:**
- **keys** (list[typing.Any])
- **values** (list[typing.Any])


# nodetool.nodes.nodetool.date

## AddTimeDelta

Add or subtract time from a datetime.

Use cases:
- Calculate future/past dates
- Generate date ranges

**Tags:** datetime, add, subtract

**Fields:**
- **input_datetime**: Starting datetime (Datetime)
- **days**: Number of days to add (negative to subtract) (int)
- **hours**: Number of hours to add (negative to subtract) (int)
- **minutes**: Number of minutes to add (negative to subtract) (int)


## DateDifference

Calculate the difference between two dates.

Use cases:
- Calculate time periods
- Measure durations

**Tags:** datetime, difference, duration

**Fields:**
- **start_date**: Start datetime (Datetime)
- **end_date**: End datetime (Datetime)


## DateFormat

## DateRange

Generate a list of dates between start and end dates.

Use cases:
- Generate date sequences
- Create date-based iterations

**Tags:** datetime, range, list

**Fields:**
- **start_date**: Start date of the range (Datetime)
- **end_date**: End date of the range (Datetime)
- **step_days**: Number of days between each date (int)


## DateToDatetime

Convert a Date object to a Datetime object.

**Tags:** date, datetime, convert

**Fields:**
- **input_date**: Date to convert (Date)


## DatetimeToDate

Convert a Datetime object to a Date object.

**Tags:** date, datetime, convert

**Fields:**
- **input_datetime**: Datetime to convert (Datetime)


## DaysAgo

Get datetime from specified days ago.

**Tags:** datetime, past, days

**Fields:**
- **days**: Number of days ago (int)


## DaysFromNow

Get datetime specified days in the future.

**Tags:** datetime, future, days

**Fields:**
- **days**: Number of days in the future (int)


## EndOfDay

Get the datetime set to the end of the day (23:59:59).

**Tags:** datetime, day, end

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## EndOfMonth

Get the datetime set to the last day of the month.

**Tags:** datetime, month, end

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## EndOfWeek

Get the datetime set to the last day of the week (Sunday by default).

**Tags:** datetime, week, end

**Fields:**
- **input_datetime**: Input datetime (Datetime)
- **start_monday**: Consider Monday as start of week (False for Sunday) (bool)


## EndOfYear

Get the datetime set to the last day of the year.

**Tags:** datetime, year, end

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## FormatDateTime

Convert a datetime object to a formatted string.

Use cases:
- Standardize date formats
- Prepare dates for different systems

**Tags:** datetime, format, convert

**Fields:**
- **input_datetime**: Datetime object to format (Datetime)
- **output_format**: Desired output format (DateFormat)


## GetQuarter

Get the quarter number and start/end dates for a given datetime.

Use cases:
- Financial reporting periods
- Quarterly analytics

**Tags:** datetime, quarter, period

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## GetWeekday

Get the weekday name or number from a datetime.

Use cases:
- Get day names for scheduling
- Filter events by weekday

**Tags:** datetime, weekday, name

**Fields:**
- **input_datetime**: Input datetime (Datetime)
- **as_name**: Return weekday name instead of number (0-6) (bool)


## HoursAgo

Get datetime from specified hours ago.

**Tags:** datetime, past, hours

**Fields:**
- **hours**: Number of hours ago (int)


## HoursFromNow

Get datetime specified hours in the future.

**Tags:** datetime, future, hours

**Fields:**
- **hours**: Number of hours in the future (int)


## IsDateInRange

Check if a date falls within a specified range.

Use cases:
- Validate date ranges
- Filter date-based data

**Tags:** datetime, range, check

**Fields:**
- **check_date**: Date to check (Datetime)
- **start_date**: Start of date range (Datetime)
- **end_date**: End of date range (Datetime)
- **inclusive**: Include start and end dates in range (bool)


## MonthsAgo

Get datetime from specified months ago.

**Tags:** datetime, past, months

**Fields:**
- **months**: Number of months ago (int)


## MonthsFromNow

Get datetime specified months in the future.

**Tags:** datetime, future, months

**Fields:**
- **months**: Number of months in the future (int)


## Now

Get the current date and time.

**Tags:** datetime, current, now

**Fields:**


## ParseDate

Parse a date string into components.

**Tags:** date, parse, format

**Fields:**
- **date_string**: The date string to parse (str)
- **input_format**: Format of the input date string (DateFormat)


## ParseDateTime

Parse a date/time string into components.

Use cases:
- Extract date components from strings
- Convert between date formats

**Tags:** datetime, parse, format

**Fields:**
- **datetime_string**: The datetime string to parse (str)
- **input_format**: Format of the input datetime string (DateFormat)


## StartOfDay

Get the datetime set to the start of the day (00:00:00).

**Tags:** datetime, day, start

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## StartOfMonth

Get the datetime set to the first day of the month.

**Tags:** datetime, month, start

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## StartOfWeek

Get the datetime set to the first day of the week (Monday by default).

**Tags:** datetime, week, start

**Fields:**
- **input_datetime**: Input datetime (Datetime)
- **start_monday**: Consider Monday as start of week (False for Sunday) (bool)


## StartOfYear

Get the datetime set to the first day of the year.

**Tags:** datetime, year, start

**Fields:**
- **input_datetime**: Input datetime (Datetime)


## Today

Get the current date.

**Tags:** date, today, now

**Fields:**


# nodetool.nodes.nodetool.json

## BaseGetJSONPath

Base class for extracting typed data from a JSON object using a path expression.

Examples for an object {"a": {"b": {"c": 1}}}
"a.b.c" -> 1
"a.b" -> {"c": 1}
"a" -> {"b": {"c": 1}}

Use cases:
- Navigate complex JSON structures
- Extract specific values from nested JSON with type safety

**Tags:** json, path, extract

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)


## FilterJSON

Filter JSON array based on a key-value condition.

Use cases:
- Filter arrays of objects
- Search JSON data

**Tags:** json, filter, array

**Fields:**
- **array**: Array of JSON objects to filter (list[dict])
- **key**: Key to filter on (str)
- **value**: Value to match (Any)


## GetJSONPathBool

Extract a boolean value from a JSON path

**Tags:** json, path, extract, boolean

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (bool)


## GetJSONPathDict

Extract a dictionary value from a JSON path

**Tags:** json, path, extract, object

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (dict)


## GetJSONPathFloat

Extract a float value from a JSON path

**Tags:** json, path, extract, number

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (float)


## GetJSONPathInt

Extract an integer value from a JSON path

**Tags:** json, path, extract, number

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (int)


## GetJSONPathList

Extract a list value from a JSON path

**Tags:** json, path, extract, array

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (list)


## GetJSONPathStr

Extract a string value from a JSON path

**Tags:** json, path, extract, string

**Fields:**
- **data**: JSON object to extract from (Any)
- **path**: Path to the desired value (dot notation) (str)
- **default** (str)


## JSONTemplate

Template JSON strings with variable substitution.

Example:
template: '{"name": "$user", "age": $age}'
values: {"user": "John", "age": 30}
result: '{"name": "John", "age": 30}'

Use cases:
- Create dynamic JSON payloads
- Generate JSON with variable data
- Build API request templates

**Tags:** json, template, substitute, variables

**Fields:**
- **template**: JSON template string with $variable placeholders (str)
- **values**: Dictionary of values to substitute into the template (dict[str, typing.Any])


## ParseDict

Parse a JSON string into a Python dictionary.

Use cases:
- Convert JSON API responses to Python dictionaries
- Process JSON configuration files
- Parse object-like JSON data

**Tags:** json, parse, decode, dictionary

**Fields:**
- **json_string**: JSON string to parse into a dictionary (str)


## ParseList

Parse a JSON string into a Python list.

Use cases:
- Convert JSON array responses to Python lists
- Process JSON data collections
- Parse array-like JSON data

**Tags:** json, parse, decode, array, list

**Fields:**
- **json_string**: JSON string to parse into a list (str)


## StringifyJSON

Convert a Python object to a JSON string.

Use cases:
- Prepare data for API requests
- Save data in JSON format

**Tags:** json, stringify, encode

**Fields:**
- **data**: Data to convert to JSON (Any)
- **indent**: Number of spaces for indentation (int)


## ValidateJSON

Validate JSON data against a schema.

Use cases:
- Ensure API payloads match specifications
- Validate configuration files

**Tags:** json, validate, schema

**Fields:**
- **data**: JSON data to validate (Any)
- **schema**: JSON schema for validation (dict)


# nodetool.nodes.nodetool.video

## AddAudio

Add an audio track to a video, replacing or mixing with existing audio.

Use cases:
1. Add background music or narration to a silent video
2. Replace original audio with a new soundtrack
3. Mix new audio with existing video sound

**Tags:** video, audio, soundtrack, merge

**Fields:**
- **video**: The input video to add audio to. (VideoRef)
- **audio**: The audio file to add to the video. (AudioRef)
- **volume**: Volume adjustment for the added audio. 1.0 is original volume. (float)
- **mix**: If True, mix new audio with existing. If False, replace existing audio. (bool)


## AddSubtitles

Add subtitles to a video.

Use cases:
1. Add translations or closed captions to videos
2. Include explanatory text or commentary in educational videos
3. Create lyric videos for music content

**Tags:** video, subtitles, text, caption

**Fields:**
- **video**: The input video to add subtitles to. (VideoRef)
- **chunks**: Audio chunks to add as subtitles. (list[nodetool.metadata.types.AudioChunk])
- **font**: The font to use. (SubtitleTextFont)
- **align**: Vertical alignment of subtitles. (SubtitleTextAlignment)
- **font_size**: The font size. (int)
- **font_color**: The font color. (ColorRef)


## Blur

Apply a blur effect to a video.

Use cases:
1. Create a dreamy or soft focus effect
2. Obscure or censor specific areas of the video
3. Reduce noise or grain in low-quality footage

**Tags:** video, blur, smooth, soften

**Fields:**
- **video**: The input video to apply blur effect. (VideoRef)
- **strength**: The strength of the blur effect. Higher values create a stronger blur. (float)


## ChromaKey

Apply chroma key (green screen) effect to a video.

Use cases:
1. Remove green or blue background from video footage
2. Create special effects by compositing video onto new backgrounds
3. Produce professional-looking videos for presentations or marketing

**Tags:** video, chroma key, green screen, compositing

**Fields:**
- **video**: The input video to apply chroma key effect. (VideoRef)
- **key_color**: The color to key out (e.g., '#00FF00' for green). (ColorRef)
- **similarity**: Similarity threshold for the key color. (float)
- **blend**: Blending of the keyed area edges. (float)


## ColorBalance

Adjust the color balance of a video.

Use cases:
1. Correct color casts in video footage
2. Enhance specific color tones for artistic effect
3. Normalize color balance across multiple video clips

**Tags:** video, color, balance, adjustment

**Fields:**
- **video**: The input video to adjust color balance. (VideoRef)
- **red_adjust**: Red channel adjustment factor. (float)
- **green_adjust**: Green channel adjustment factor. (float)
- **blue_adjust**: Blue channel adjustment factor. (float)


## Concat

Concatenate multiple video files into a single video, including audio when available.

**Tags:** video, concat, merge, combine, audio, +

**Fields:**
- **video_a**: The first video to concatenate. (VideoRef)
- **video_b**: The second video to concatenate. (VideoRef)


## CreateVideo

Combine a sequence of frames into a single video file.

Use cases:
1. Create time-lapse videos from image sequences
2. Compile processed frames back into a video
3. Generate animations from individual images

**Tags:** video, frames, combine, sequence

**Fields:**
- **frames**: The frames to combine into a video. (list[nodetool.metadata.types.ImageRef])
- **fps**: The FPS of the output video. (float)


## Denoise

Apply noise reduction to a video.

Use cases:
1. Improve video quality by reducing unwanted noise
2. Enhance low-light footage
3. Prepare video for further processing or compression

**Tags:** video, denoise, clean, enhance

**Fields:**
- **video**: The input video to denoise. (VideoRef)
- **strength**: Strength of the denoising effect. Higher values mean more denoising. (float)


## ExtractAudio

Separate audio from a video file.

**Tags:** video, audio, extract, separate

**Fields:**
- **video**: The input video to separate. (VideoRef)


## ExtractFrames

Extract frames from a video file using OpenCV.

Use cases:
1. Generate image sequences for further processing
2. Extract specific frame ranges from a video
3. Create thumbnails or previews from video content

**Tags:** video, frames, extract, sequence

**Fields:**
- **video**: The input video to extract frames from. (VideoRef)
- **start**: The frame to start extracting from. (int)
- **end**: The frame to stop extracting from. (int)

### result_for_client

**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]


## Fps

Get the frames per second (FPS) of a video file.

Use cases:
1. Analyze video properties for quality assessment
2. Determine appropriate playback speed for video editing
3. Ensure compatibility with target display systems

**Tags:** video, analysis, frames, fps

**Fields:**
- **video**: The input video to analyze for FPS. (VideoRef)


## Overlay

Overlay one video on top of another, including audio overlay when available.

**Tags:** video, overlay, composite, picture-in-picture, audio

**Fields:**
- **main_video**: The main (background) video. (VideoRef)
- **overlay_video**: The video to overlay on top. (VideoRef)
- **x**: X-coordinate for overlay placement. (int)
- **y**: Y-coordinate for overlay placement. (int)
- **scale**: Scale factor for the overlay video. (float)
- **overlay_audio_volume**: Volume of the overlay audio relative to the main audio. (float)


## ResizeNode

Resize a video to a specific width and height.

Use cases:
1. Adjust video resolution for different display requirements
2. Reduce file size by downscaling video
3. Prepare videos for specific platforms with size constraints

**Tags:** video, resize, scale, dimensions

**Fields:**
- **video**: The input video to resize. (VideoRef)
- **width**: The target width. Use -1 to maintain aspect ratio. (int)
- **height**: The target height. Use -1 to maintain aspect ratio. (int)


## Reverse

Reverse the playback of a video.

Use cases:
1. Create artistic effects by playing video in reverse
2. Analyze motion or events in reverse order
3. Generate unique transitions or intros for video projects

**Tags:** video, reverse, backwards, effect

**Fields:**
- **video**: The input video to reverse. (VideoRef)


## Rotate

Rotate a video by a specified angle.

Use cases:
1. Correct orientation of videos taken with a rotated camera
2. Create artistic effects by rotating video content
3. Adjust video for different display orientations

**Tags:** video, rotate, orientation, transform

**Fields:**
- **video**: The input video to rotate. (VideoRef)
- **angle**: The angle of rotation in degrees. (float)


## Saturation

Adjust the color saturation of a video.

Use cases:
1. Enhance color vibrancy in dull or flat-looking footage
2. Create stylistic effects by over-saturating or desaturating video
3. Correct oversaturated footage from certain cameras

**Tags:** video, saturation, color, enhance

**Fields:**
- **video**: The input video to adjust saturation. (VideoRef)
- **saturation**: Saturation level. 1.0 is original, <1 decreases saturation, >1 increases saturation. (float)


## SaveVideo

Save a video to a file.

Use cases:
1. Export processed video to a specific folder
2. Save video with a custom name
3. Create a copy of a video in a different location

**Tags:** video, save, file, output

**Fields:**
- **video**: The video to save. (VideoRef)
- **folder**: Name of the output folder. (FolderRef)
- **name**: 
        Name of the output video.
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


## SetSpeed

Adjust the playback speed of a video.

Use cases:
1. Create slow-motion effects by decreasing video speed
2. Generate time-lapse videos by increasing playback speed
3. Synchronize video duration with audio or other timing requirements

**Tags:** video, speed, tempo, time

**Fields:**
- **video**: The input video to adjust speed. (VideoRef)
- **speed_factor**: The speed adjustment factor. Values > 1 speed up, < 1 slow down. (float)


## Sharpness

Adjust the sharpness of a video.

Use cases:
1. Enhance detail in slightly out-of-focus footage
2. Correct softness introduced by video compression
3. Create stylistic effects by over-sharpening

**Tags:** video, sharpen, enhance, detail

**Fields:**
- **video**: The input video to sharpen. (VideoRef)
- **luma_amount**: Amount of sharpening to apply to luma (brightness) channel. (float)
- **chroma_amount**: Amount of sharpening to apply to chroma (color) channels. (float)


## Stabilize

Apply video stabilization to reduce camera shake and jitter.

Use cases:
1. Improve quality of handheld or action camera footage
2. Smooth out panning and tracking shots
3. Enhance viewer experience by reducing motion sickness

**Tags:** video, stabilize, smooth, shake-reduction

**Fields:**
- **video**: The input video to stabilize. (VideoRef)
- **smoothing**: Smoothing strength. Higher values result in smoother but potentially more cropped video. (float)
- **crop_black**: Whether to crop black borders that may appear after stabilization. (bool)


## Transition

Create a transition effect between two videos, including audio transition when available.

Use cases:
1. Create smooth transitions between video clips in a montage
2. Add professional-looking effects to video projects
3. Blend scenes together for creative storytelling
4. Smoothly transition between audio tracks of different video clips

**Tags:** video, transition, effect, merge, audio

**Fields:**
- **video_a**: The first video in the transition. (VideoRef)
- **video_b**: The second video in the transition. (VideoRef)
- **transition_type**: Type of transition effect (TransitionType)
- **duration**: Duration of the transition effect in seconds. (float)


## Trim

Trim a video to a specific start and end time.

Use cases:
1. Extract specific segments from a longer video
2. Remove unwanted parts from the beginning or end of a video
3. Create shorter clips from a full-length video

**Tags:** video, trim, cut, segment

**Fields:**
- **video**: The input video to trim. (VideoRef)
- **start_time**: The start time in seconds for the trimmed video. (float)
- **end_time**: The end time in seconds for the trimmed video. Use -1 for the end of the video. (float)


### safe_unlink

**Args:**
- **path (str)**

# nodetool.nodes.nodetool.image

## BatchToList

Convert an image batch to a list of image references.

Use cases:
- Convert comfy batch outputs to list format

**Tags:** batch, list, images, processing

**Fields:**
- **batch**: The batch of images to convert. (ImageRef)


## Crop

Crop an image to specified coordinates.

- Remove unwanted borders from images
- Focus on particular subjects within an image
- Simplify images by removing distractions

**Tags:** image, crop

**Fields:**
- **image**: The image to crop. (ImageRef)
- **left**: The left coordinate. (int)
- **top**: The top coordinate. (int)
- **right**: The right coordinate. (int)
- **bottom**: The bottom coordinate. (int)


## Fit

Resize an image to fit within specified dimensions while preserving aspect ratio.

- Resize images for online publishing requirements
- Preprocess images to uniform sizes for machine learning
- Control image display sizes for web development

**Tags:** image, resize, fit

**Fields:**
- **image**: The image to fit. (ImageRef)
- **width**: Width to fit to. (int)
- **height**: Height to fit to. (int)


## GetMetadata

Get metadata about the input image.

Use cases:
- Use width and height for layout calculations
- Analyze image properties for processing decisions
- Gather information for image cataloging or organization

**Tags:** metadata, properties, analysis, information

**Fields:**
- **image**: The input image. (ImageRef)


## Paste

Paste one image onto another at specified coordinates.

Use cases:
- Add watermarks or logos to images
- Combine multiple image elements
- Create collages or montages

**Tags:** paste, composite, positioning, overlay

**Fields:**
- **image**: The image to paste into. (ImageRef)
- **paste**: The image to paste. (ImageRef)
- **left**: The left coordinate. (int)
- **top**: The top coordinate. (int)


## Resize

Change image dimensions to specified width and height.

- Preprocess images for machine learning model inputs
- Optimize images for faster web page loading
- Create uniform image sizes for layouts

**Tags:** image, resize

**Fields:**
- **image**: The image to resize. (ImageRef)
- **width**: The target width. (int)
- **height**: The target height. (int)


## SaveImage

Save an image to specified folder with customizable name format.

Use cases:
- Save generated images with timestamps
- Organize outputs into specific folders
- Create backups of processed images

**Tags:** save, image, folder, naming

**Fields:**
- **image**: The image to save. (ImageRef)
- **folder**: The folder to save the image in. (FolderRef)
- **name**: 
        Name of the output file.
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

### result_for_client

**Args:**
- **result (dict[str, typing.Any])**

**Returns:** dict[str, typing.Any]


## Scale

Enlarge or shrink an image by a scale factor.

- Adjust image dimensions for display galleries
- Standardize image sizes for machine learning datasets
- Create thumbnail versions of images

**Tags:** image, resize, scale

**Fields:**
- **image**: The image to scale. (ImageRef)
- **scale**: The scale factor. (float)


# nodetool.nodes.nodetool.output

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



