# CLAUDE.md - AI Assistant Guidelines

This file provides guidance for AI assistants (like Claude) working with this repository.

## Repository Overview

**Repository:** claude-code-web-test
**Owner:** yunho0130
**Purpose:** Test repository for Claude Code web interface functionality
**Status:** New/Initial setup

## Project Structure

```
claude-code-web-test/
├── README.md          # Project description
├── CLAUDE.md          # This file - AI assistant guidelines
└── .git/              # Git version control
```

## Development Workflow

### Git Branching Strategy

- **Main branch:** Primary stable branch
- **Feature branches:** Use `claude/` prefix for AI-assisted development
- Commit messages should be clear and descriptive
- Push changes using: `git push -u origin <branch-name>`

### Making Changes

1. Always read files before modifying them
2. Use clear, descriptive commit messages
3. Keep changes focused and minimal
4. Test changes when applicable before committing

## Code Conventions

### General Guidelines

- Keep code simple and readable
- Avoid over-engineering
- Follow existing patterns in the codebase
- Document non-obvious logic with comments

### File Naming

- Use lowercase with hyphens for file names (e.g., `my-file.js`)
- Use descriptive names that reflect file contents

## AI Assistant Instructions

### When Working on This Repository

1. **Explore first:** Read existing files to understand context before making changes
2. **Minimal changes:** Only modify what's necessary for the task
3. **No assumptions:** Ask for clarification when requirements are unclear
4. **Track progress:** Use task tracking for multi-step operations
5. **Clean commits:** Create atomic, well-described commits

### Common Tasks

- **Adding features:** Create necessary files, follow existing patterns
- **Bug fixes:** Identify root cause, fix minimally, verify fix
- **Documentation:** Keep docs up-to-date with code changes

### What to Avoid

- Creating unnecessary files or abstractions
- Adding features not explicitly requested
- Making changes without reading existing code first
- Over-commenting or adding redundant documentation

## Testing

Currently no test framework is configured. When tests are added:
- Run tests before committing changes
- Ensure all tests pass before pushing

## Build & Deploy

No build process currently configured. Update this section when build tooling is added.

## Dependencies

No dependencies currently. Update this section when package management is configured.

---

*Last updated: 2025-12-01*
