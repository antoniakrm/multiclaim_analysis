import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from llm_inference.engine import Engine
from llm_inference.schema import schema_registry
from llm_inference.utils import TsvWriter, setup_config, setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig):
    setup_logging()
    setup_config(config)

    engine = Engine(
        llm=instantiate(config.model),
        sampling_params=instantiate(config.generation),
        schema=schema_registry.get(config.schema_name),
        dataset=instantiate(config.data),
    )

    with TsvWriter(Path(config.output_path)) as writer:
        for i, json_completion in enumerate(engine()):
            if i == 0:
                writer.writerow(json_completion.keys_as_list())
            writer.writerow(json_completion.values_as_list())

            if config.log_outputs:
                logger.info(f"{i}: {json_completion.output_json}")


if __name__ == "__main__":
    main()
