from toolsgen import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
)

gen_config = GenerationConfig(
    num_samples=10_000,
    strategy="random",
    seed=42,
    train_split=0.8,
    language="english",
    max_attempts=3,
    k_min=2,
    k_max=8,
    shuffle_tools=True,
    num_workers=8,
    worker_batch_size=16,
)

role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="gpt-5-mini",
        temperature=1.0,
    ),
    tool_caller=ModelConfig(
        model="gpt-5-mini",
        temperature=0,
    ),
    judge=ModelConfig(
        model="gpt-5",
        temperature=0,
    ),
)
