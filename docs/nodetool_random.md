---
layout: default
title: nodetool.random
parent: Nodes
has_children: false
nav_order: 2
---

# nodetool.nodes.nodetool.random

Random utilities for generating random data.

## RandomBool

Return a random boolean value.

Use cases:
- Make random yes/no decisions
- Simulate coin flips
- Introduce randomness in control flow

**Tags:** random, boolean, coinflip, bool

**Fields:**


## RandomChoice

Select a random element from a list.

Use cases:
- Choose a random sample from options
- Implement simple lottery behaviour
- Pick a random item from user input

**Tags:** random, choice, select, pick

**Fields:**
- **options** (list[typing.Any])


## RandomFloat

Generate a random floating point number within a range.

Use cases:
- Create random probabilities
- Generate noisy data for testing
- Produce random values for simulations

**Tags:** random, float, number, rand, uniform

**Fields:**
- **minimum** (float)
- **maximum** (float)


## RandomInt

Generate a random integer within a range.

Use cases:
- Pick a random index or identifier
- Create randomized counters or IDs
- Sample integers for testing

**Tags:** random, integer, number, rand, randint

**Fields:**
- **minimum** (int)
- **maximum** (int)
