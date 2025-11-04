# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Project skeleton with package, CLI, tests, and basic schemas
- Core dataset generation functionality
  - LLM-based user request generation
  - Tool call generation with OpenAI-compatible APIs
  - Record creation and serialization
- Complete CLI implementation with model configuration options
- Comprehensive test suite for schemas, sampling, config, and generator
- Public API exports in `__init__.py`
- Support for `param_aware` and `semantic` sampling strategies
- LLM-as-a-judge scoring system
  - Rubric-based evaluation (tool relevance, argument quality, clarity)
  - Structured outputs using pydantic models for reliable parsing
- JSONL output format with train/val split support
- Prompts module with centralized prompt templates
- Semantic sampling strategy based on keyword similarity

### Changed
- **[BREAKING]** Reduced dependencies (removed typer, python-dotenv from core dependencies)
- **[BREAKING]** Flattened module structure - removed nested folders (io/, judge/, providers/)
- **[BREAKING]** CLI rewritten using `argparse` instead of `typer` (stdlib only)
- **[BREAKING]** Module reorganization:
  - `generator.py`, `config.py`, `io/writer.py` merged into `core.py`
  - `judge/scorer.py` moved to `judge.py`
  - `providers/openai_compat.py` removed - using OpenAI SDK directly
- **[BREAKING]** Import paths updated:
  - `from toolsgen.config` → `from toolsgen.core`
  - `from toolsgen.generator` → `from toolsgen.core`
  - `from toolsgen.judge.scorer` → `from toolsgen.judge`
- Simplified semantic sampling algorithm (reduced complexity)
- Environment variable loading: direct `os.environ` instead of python-dotenv for core functionality
- Enhanced error handling and user feedback in CLI
- Output format uses JSONL for datasets
- Judge system uses OpenAI SDK structured outputs directly
- Prompts extracted from inline strings to dedicated module
- Updated README with simplified installation and new import paths

### Removed
- Dependency on `typer` (replaced with stdlib `argparse`)
- Dependency on `python-dotenv` from core (moved to dev dependencies, used optionally in examples)
- OpenAI wrapper abstraction layer (use SDK directly)
- Nested folder structure (io/, judge/, providers/)

### Fixed
- Missing dependencies declaration in `pyproject.toml`
- Reduced code complexity and improved maintainability
