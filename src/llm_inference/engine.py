from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Generator, List, Tuple, Type, Union

from outlines.serve.vllm import JSONLogitsProcessor
from pydantic import BaseModel
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .dataset import ClaimPostDataset
from .utils import parse_json, parse_pydantic_schema, validate_json_with_schema


@dataclass
class JsonCompletion:
    input_json: Dict[str, Any]
    output_json: Dict[str, Any]

    def values_as_list(self) -> List[Any]:
        return list(self.input_json.values()) + list(self.output_json.values())

    def keys_as_list(self) -> List[str]:
        return list(self.input_json.keys()) + list(self.output_json.keys())


class Engine:
    def __init__(
        self,
        llm: LLM,
        sampling_params: SamplingParams,
        schema: Type[BaseModel],
        dataset: ClaimPostDataset,
    ):
        self.llm = llm
        self.sampling_params = sampling_params
        self.schema = schema
        self.dataset = dataset

        self.sampling_params.logits_processors = [
            JSONLogitsProcessor(schema=self.schema, llm=self.llm, whitespace_pattern=r" ?")
        ]

    @cached_property
    def schema_string(self) -> str:
        return parse_pydantic_schema(self.schema)

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[JsonCompletion]:
        json_completions = [JsonCompletion(input_json=ex, output_json={}) for ex in batch]

        queue = deque([(i, ex) for i, ex in enumerate(batch)])

        while queue:
            indices = []
            prompts = []
            while queue:
                i, example = queue.popleft()
                indices.append(i)
                prompts.append(self.dataset.template.render(**example, schema=self.schema_string))

            raw_outputs = self.llm.generate(
                prompts, sampling_params=self.sampling_params, use_tqdm=False
            )

            for i, raw_output in zip(indices, raw_outputs):
                json_output = parse_json(raw_output.outputs[0].text)
                if not validate_json_with_schema(json_output, self.schema):
                    queue.append((i, batch[i]))
                else:
                    json_completions[i].output_json = json_output

        return json_completions

    def __call__(self) -> Generator[JsonCompletion, None, None]:
        for batch in tqdm(self.dataset, desc="Processing batches"):
            yield from self.process_batch(batch)
