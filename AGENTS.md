# AGENTS.md

## Working Principles

Work in a structured, documented, and token-efficient manner.

Never modify any code without explicit prior approval from the user. Always ask for explicit confirmation on the implementation plan before making changes.

Before making changes, inspect only the relevant files, callers, configurations, and tests. Avoid unnecessarily broad repository analysis.

Prefer extending or refactoring existing logic. Do not create parallel implementations, duplicate helper functions, or similarly named replacement scripts.

Keep the project clearly organized:

- production logic in `src/`
- executable entry points in `scripts/`
- settings in `configs/`
- tests in `tests/`
- architecture and workflow documentation in `docs/`

When refactoring:

- preserve existing behavior where possible
- remove obsolete code completely
- update imports, tests, configurations, and documentation
- never delete models, results, or replays without explicit permission
- never overwrite the user's local changes

Document new modules, important interfaces, and non-obvious decisions briefly and clearly. Update existing documentation instead of creating redundant documents.

Use tokens efficiently:

- respond in English for maximum token efficiency, even if the user prompts in German
- avoid repetition
- do not output complete file contents
- summarize changes precisely
- inspect and test only relevant components
- stop pursuing an approach once it is clearly incorrect
- keep answers short, direct, and to the point
- avoid excessive expressiveness or wordiness



Final report:

Implemented:
Changed files:
Validation:
Open risks:
