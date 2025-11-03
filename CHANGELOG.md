# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Project skeleton with package, CLI, tests, and basic schemas
- Core dataset generation functionality (`generator.py`)
  - LLM-based user request generation
  - Tool call generation with OpenAI-compatible APIs
  - Record creation and serialization
- Complete CLI implementation with model configuration options
- Comprehensive test suite for schemas, sampling, config, and generator
- Public API exports in `__init__.py`
- Support for `param_aware` and `semantic` sampling strategies
- LLM-as-a-judge scoring system (`judge/scorer.py`)
  - Rubric-based evaluation (tool relevance, argument quality, clarity)
  - Structured outputs using pydantic models for reliable parsing
- JSONL output format (`io/writer.py`)
  - Train/val split support
- Prompts module (`prompts.py`) with centralized prompt templates
- OpenAI adapter enhancements:
  - Retry logic with exponential backoff
  - Rate limiting using token bucket algorithm
  - Structured outputs support via `create_structured()` method
- Semantic sampling strategy based on keyword similarity

### Changed
- CLI `generate` command now performs actual dataset generation
- Updated dependencies in `pyproject.toml` to include all required packages
- Enhanced error handling and user feedback in CLI
- Output format changed from individual JSON files to JSONL
- Judge prompt uses structured outputs instead of manual JSON parsing
- Prompts extracted from inline strings to dedicated module

### Fixed
- Missing dependencies declaration in `pyproject.toml`
