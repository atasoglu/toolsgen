# Nano Tool Calling v1

Generates 10K tool-calling samples from Hugging Face dataset with parallel processing and schema validation.

## Setup

```bash
pip install toolsgen datasets python-dotenv
echo "OPENAI_API_KEY=your-key-here" > .env
python example.py
```

## Configuration

- **Dataset**: `argilla-warehouse/python-seed-tools`
- **Samples**: 10,000 (80% train / 20% val)
- **Parallel**: 8 workers × 16 batch size
- **Models**: GPT-4.1-nano

## Files

- `example.py` - Main generation script
- `config.py` - Generation and model settings
- `utils.py` - HF dataset loader
- `validation.py` - Schema validator (ensures arrays have `items`)

## Output

- `output/train.jsonl` - Training set
- `output/val.jsonl` - Validation set
- `output/manifest.json` - Metadata
