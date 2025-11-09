from toolsgen import (
    GenerationConfig,
    ModelConfig,
    RoleBasedModelConfig,
)

gen_config = GenerationConfig(
    num_samples=1_000,
    strategy="random",
    seed=42,
    language="english",
    max_attempts=3,
    k_min=1,
    k_max=4,
    shuffle_tools=True,
    num_workers=16,
    worker_batch_size=16,
)

role_config = RoleBasedModelConfig(
    problem_generator=ModelConfig(
        model="gpt-4.1-nano",
        temperature=1.0,
    ),
    tool_caller=ModelConfig(
        model="gpt-4.1-nano",
        temperature=0,
    ),
    judge=ModelConfig(
        model="gpt-4.1-mini",
        temperature=0,
    ),
)
