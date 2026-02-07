# Claude Memory

This directory contains context and instructions for the Claude Code autonomous agent that maintains Kie.ai nodes in this repository.

## Purpose

The agent periodically scans the Kie.ai marketplace to discover new models and creates corresponding NodeTool nodes for them.

## Files

- `README.md` - This file
- `repository-context.md` - Overview of the repository structure and key paths
- `node-creation-guide.md` - Step-by-step instructions for creating new Kie.ai nodes
- `features.md` - Log of features added by the agent

## Related Workflows

- `.github/workflows/claude-kie-sync.yml` - Scheduled workflow that runs the agent
