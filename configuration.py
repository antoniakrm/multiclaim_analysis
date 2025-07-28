from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    _target_: str = "vllm.LLM"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int = 2048


@dataclass
class GenerationConfig:
    _target_: str = "vllm.SamplingParams"
    max_tokens: int = 200
    temperature: float = 0.4
    top_k: int = 50
    top_p: float = 1.0


@dataclass
class DataConfig:
    _target_: str = "llm_inference.dataset.ClaimPostDataset"
    fact_checks_csv: str = MISSING
    posts_csv: str = MISSING
    mapping_csv: str = MISSING
    retrieval_json: str = MISSING
    template_name: str = "llama3_template"
    batch_size: int = 1


@dataclass
class RunConfig:
    output_path: str = MISSING
    model: ModelConfig = ModelConfig()
    generation: GenerationConfig = GenerationConfig()
    data: DataConfig = DataConfig()
    schema_name: str = "relevance_check"
    log_outputs: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=RunConfig)
