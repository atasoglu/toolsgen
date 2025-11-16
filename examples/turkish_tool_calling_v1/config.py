from random import randint
from toolsgen import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
)

seed = randint(1, 10_000_000)
print(f"Using seed: {seed}")

openai_params = dict(
    base_url="https://openrouter.ai/api/v1",
)

gen_config = GenerationConfig(
    num_samples=1000,
    strategy="random",
    seed=seed,
    train_split=0.8,
    language="turkish",
    max_attempts=1,
    k_min=1,
    k_max=8,
    shuffle_tools=True,
)

role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=1.0,
        openai_params=openai_params,
        max_tokens=500,
    ),
    tool_caller=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=0,
        openai_params=openai_params,
        max_tokens=500,
    ),
    judge=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=0,
        openai_params=openai_params,
        max_tokens=500,
    ),
)
