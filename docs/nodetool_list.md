---
layout: default
title: nodetool.list
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.list

Utilities for manipulating lists: filtering, sorting and generating sequences.

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


