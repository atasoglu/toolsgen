from toolsgen import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
)

openai_params = dict(
    base_url="https://openrouter.ai/api/v1",
)

gen_config = GenerationConfig(
    num_samples=...,
    strategy="random",
    seed=...,
    train_split=0.8,
    language="turkish",
    max_attempts=1,
    k_min=1,
    k_max=8,
    shuffle_tools=True,
    num_workers=16,
    worker_batch_size=1,
)

role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=1.0,
        openai_params=openai_params,
    ),
    tool_caller=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=0,
        openai_params=openai_params,
    ),
    judge=ModelConfig(
        model="qwen/qwen3-235b-a22b-2507",
        temperature=0,
        openai_params=openai_params,
    ),
)
