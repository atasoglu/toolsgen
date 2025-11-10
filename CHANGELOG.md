# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

Nothing yet.

## [0.5.0] - 2025-01-11
### Added
- Hugging Face Hub integration for direct dataset uploads
  - `push_to_hub()` function in new `hf_hub` module to upload datasets to HF Hub
  - Uploads JSONL files (train.jsonl, val.jsonl), manifest.json, and auto-generated README.md
  - CLI flags: `--push-to-hub`, `--repo-id`, `--hf-token`, `--private`
  - Support for both public and private repositories
  - Auto-generated dataset cards with dataset statistics, model info, usage examples, and citation
- Optional dependency: `huggingface_hub>=0.20.0` (install with `pip install toolsgen[hf]`)
- Example in `examples/hf_hub_upload/` with dotenv configuration
- Test suite for HF Hub functionality in `tests/test_hf_hub.py`
- `push_to_hub` exported from main `toolsgen` package for easier imports

## [0.4.0] - 2025-01-10
### Added
- Quality tagging system for generated records
  - `generate_quality_tags()` method in `JudgeResponse` to automatically tag samples based on judge scores
  - Tags include overall quality levels (high/medium/low_quality) and dimension-specific tags (excellent/poor tool selection, arguments, clarity)
  - Configurable thresholds for quality classification
  - `quality_tags` field automatically populated in generated records

## [0.3.0] - 2025-01-10
### Added
- Hugging Face dataset integration utilities in `examples/nano_tool_calling_v1/`
  - `dataset_to_tools()` function to load tools from Hugging Face datasets
  - `validate_json_schema()` for OpenAI tool schema validation with recursive array type checking
  - `push_to_hf.py` script for uploading generated datasets to Hugging Face Hub
- Complete example workflow for Nano Tool Calling v1 dataset generation
  - Configuration, generation, validation, and publishing pipeline
  - Analysis utilities for function inspection
  - Comprehensive README with dataset card format

### Changed
- Enhanced batch sampling progress bar display for better user feedback
- Improved parallel processing record ordering and ID assignment

## [0.2.2] - 2025-01-09
### Changed
- Records are now written to JSONL file immediately as they complete in parallel mode, rather than waiting for all generation to finish
- Improved memory efficiency by removing records from buffer after writing to disk

## [0.2.1] - 2025-01-09
### Fixed
- Fixed integration tests to work with refactored module structure

## [0.2.0] - 2025-01-09
### Added
- Parallel generation support with multiprocessing via `--workers` and `--worker-batch-size` CLI flags
- `num_workers` and `worker_batch_size` configuration options in `GenerationConfig`
- Parallel generation example in `examples/parallel/`

### Fixed
- Fixed tool subset diversity preservation in parallel mode by sorting records by original sample index before assigning final IDs

## [0.1.4] - 2025-11-09
### Changed
- Made `max_tokens` optional across all chat completion helpers and dataset flows so callers can rely on model defaults unless a limit is explicitly set.

## [0.1.3] - 2025-11-08
### Added
- Batching controls (`batch_size`, `shuffle_tools`) in `GenerationConfig`, CLI flags, and docs to opt into chunked sampling.
- Deterministic chunk-based sampling path that reuses batches in a wrap-around manner when generating many subsets.

### Changed
- CLI now forwards batching parameters so dataset generation can reuse the refactored sampling logic end-to-end.

## [0.1.2] - 2025-11-08
### Fixed
- Restored `toolsgen version` output by sourcing `__version__` from package metadata when running the CLI

## [0.1.1] - 2025-11-08
### Added
- Official support declarations for Python 3.12, 3.13, and 3.14 in project metadata

## [0.1.0] - 2025-11-08
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
- **[BREAKING]** `generate_dataset()` function signature updated:
  - New signature: `generate_dataset(output_dir, gen_config, model_config, tools_path=None, tools=None)`
  - Old signature: `generate_dataset(tools_path, output_dir, gen_config, model_config)`
  - Now supports passing tools list directly via `tools` parameter as alternative to `tools_path`
  - Exactly one of `tools_path` or `tools` must be provided
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
