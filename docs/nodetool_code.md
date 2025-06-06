---
layout: default
title: nodetool.code
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.code

Execute small pieces of Python code or evaluate expressions in a controlled environment.

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


