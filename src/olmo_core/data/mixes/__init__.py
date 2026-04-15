import os
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Tuple

from olmo_core.config import StrEnum

from ..tokenizer import TokenizerName

__all__ = ["DataMixBase", "DataMix"]


class DataMixBase(StrEnum):
    """
    Base class for enumeration of data mixes.
    """

    @abstractmethod
    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        """
        Construct the data mix.

        :param base_dir: Where the mix is stored, e.g. "s3://ai2-llm" or "/weka/oe-training-default/ai2-llm".
        :param tokenizer: The tokenizer identifier.

        :returns: A list of paths/URLs to the tokenized numpy data files in the mix and list
            of corresponding labels.
        """
        raise NotImplementedError


class DataMix(DataMixBase):
    """
    An enumeration of data mix names.
    """

    # Pretraining mixes
    OLMoE_mix_0824 = "OLMoE-mix-0824"
    dolma17 = "dolma17"
    OLMo_mix_0625 = "OLMo-mix-0625"
    OLMo_mix_0625_150Bsample = "OLMo-mix-0625-150Bsample"
    OLMo_dclm = "OLMo-dclm-sample"
    OLMo_synthetic = "OLMo-synthetic"
    OLMo_dclm_only = "OLMo-dclm-only"
    OLMo_dclm_chin0_05 = "OLMo-dclm-chin0_05"
    OLMo_dclm_chin0_1 = "OLMo-dclm-chin0_1"
    OLMo_dclm_chin0_25 = "OLMo-dclm-chin0_25"
    OLMo_dclm_chin0_5 = "OLMo-dclm-chin0_5"
    OLMo_dclm_para_chin0_05 = "OLMo-dclm-para-chin0_05"
    OLMo_dclm_para_chin0_1 = "OLMo-dclm-para-chin0_1"
    OLMo_dclm_para_chin0_25 = "OLMo-dclm-para-chin0_25"
    OLMo_dclm_para_chin0_5 = "OLMo-dclm-para-chin0_5"
    OLMo_dclm_para_chin1 = "OLMo-dclm-para-chin1"
    OLMo_dclm_chin1 = "OLMo-dclm-chin1"
    OLMo_dclm_chin2 = "OLMo-dclm-chin2"
    OLMo_dclm_chin4 = "OLMo-dclm-chin4"
    OLMo_dclm_chin8 = "OLMo-dclm-chin8"
    OLMo_dclm_chin16 = "OLMo-dclm-chin16"
    OLMo_synthetic_chin4 = "OLMo-synthetic-chin4"
    OLMo_synthetic_chin8 = "OLMo-synthetic-chin8"
    OLMo_synthetic_chin16 = "OLMo-synthetic-chin16"
    OLMo_synthetic_chin16_repeat4 = "OLMo-synthetic-chin16-repeat4"
    OLMo_repeat32_synthetic32 = "OLMo-repeat32-synthetic32"
    OLMo_repeat64_synthetic32 = "OLMo-repeat64-synthetic32"
    OLMo_repeat16_synthetic48 = "OLMo-repeat16-synthetic48"
    OLMo_repeat64_synthetic6 = "OLMo-repeat64-synthetic6"
    OLMo_repeat64_synthetic13 = "OLMo-repeat64-synthetic13"
    OLMo_repeat64_synthetic64 = "OLMo-repeat64-synthetic64"
    OLMo_dolma_0_03B  = "OLMo-dolma-0.03B"
    OLMo_dolma_0_06B  = "OLMo-dolma-0.06B"
    OLMo_dolma_0_12B  = "OLMo-dolma-0.12B"
    OLMo_dolma_0_15B  = "OLMo-dolma-0.15B"
    OLMo_dolma_0_3B   = "OLMo-dolma-0.3B"
    OLMo_dolma_0_37B  = "OLMo-dolma-0.37B"
    OLMo_dolma_0_6B   = "OLMo-dolma-0.6B"
    OLMo_dolma_0_74B  = "OLMo-dolma-0.74B"
    OLMo_dolma_1_2B   = "OLMo-dolma-1.2B"
    OLMo_dolma_1_85B  = "OLMo-dolma-1.85B"
    OLMo_dolma_2_4B   = "OLMo-dolma-2.4B"
    OLMo_dolma_3_7B   = "OLMo-dolma-3.7B"
    OLMo_dolma_4_8B   = "OLMo-dolma-4.8B"
    OLMo_dolma_7_4B   = "OLMo-dolma-7.4B"
    OLMo_dolma_9_6B   = "OLMo-dolma-9.6B"
    OLMo_dolma_14_8B  = "OLMo-dolma-14.8B"
    OLMo_dolma_15_2B  = "OLMo-dolma-15.2B"
    OLMo_dolma_19_2B  = "OLMo-dolma-19.2B"
    OLMo_dolma_29_6B  = "OLMo-dolma-29.6B"
    OLMo_dolma_30_4B  = "OLMo-dolma-30.4B"
    OLMo_dolma_59_2B  = "OLMo-dolma-59.2B"
    OLMo_dolma_60_8B  = "OLMo-dolma-60.8B"
    OLMo_dolma_0_19B  = "OLMo-dolma-0.19B"
    OLMo_dolma_0_38B  = "OLMo-dolma-0.38B"
    OLMo_dolma_0_95B  = "OLMo-dolma-0.95B"
    OLMo_dolma_1_9B   = "OLMo-dolma-1.9B"
    OLMo_dolma_3_8B   = "OLMo-dolma-3.8B"
    OLMo_dolma_7_6B   = "OLMo-dolma-7.6B"
    OLMo_dolma_val    = "OLMo-dolma-val"
    dclm_validation = "dclm-validation"
    OLMo_mix_0625_700Bsample = "OLMo-mix-0625-700Bsample"
    OLMo_mix_0625_official = "OLMo-mix-0625-official"
    OLMo_mix_0925 = "OLMo-mix-0925"
    OLMo_mix_0925_official = "OLMo-mix-0925-official"

    # Midtraining mixes
    OLMo_midtraining_mix_0625_100B = "OLMo-midtraining-mix-0725-100B"
    OLMo_midtraining_mix_0925_ingredient1_100B = "OLMo-midtraining-mix-0925-ingredient1-100B"
    OLMo_midtraining_mix_0925_ingredient2_100B = "OLMo-midtraining-mix-0925-ingredient2-100B"

    # Long-context extension mixes
    OLMo_longmino_mix_0625 = "OLMo-longmino-mix-0625"
    OLMo_longmino_mix_0925 = "OLMo-longmino-mix-0925"

    # Validation mixes
    v3_small_ppl_validation = "v3-small-ppl-validation"

    @classmethod
    def _missing_(cls, value: object) -> "DataMix | None":
        """Handle alias lookups."""
        # Aliases mapping
        aliases = {
            "dolma3-0625-6T-mix": "OLMo-mix-0625",
            "dolma3-0925-6T-mix": "OLMo-mix-0925",
            "dolma3-0925-150B-mix": "OLMo-mix-0625-150Bsample",
        }

        # Check if the value is an alias
        if isinstance(value, str) and value in aliases:
            # Look up the real value and return the corresponding enum member
            real_value = aliases[value]
            for member in cls:
                if member.value == real_value:
                    return member
        return None

    def build(self, base_dir: str, tokenizer: str) -> Tuple[List[str], List[str]]:
        if not base_dir.endswith("/"):
            base_dir = base_dir + "/"

        tokenizer_id: str = tokenizer
        if self == DataMix.v3_small_ppl_validation:
            if tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
                tokenizer_id = "gptneox20b"
            elif tokenizer == TokenizerName.dolma2:
                tokenizer_id = "dolma2-tokenizer"
        elif self == DataMix.OLMo_mix_0625:
            if tokenizer == TokenizerName.dolma2_sigdig:
                tokenizer_id = "dolma2-tokenizer-sigdig"
        elif self in [
            # Mixes used for OLMo3 training are saved with "dolma3-tokenizer" tokenizer,
            # which is exactly the same as "dolma2-tokenizer" but with a different name.
            DataMix.OLMo_mix_0625_official,
            DataMix.OLMo_mix_0925_official,
            DataMix.OLMo_midtraining_mix_0625_100B,
            DataMix.OLMo_midtraining_mix_0925_ingredient1_100B,
            DataMix.OLMo_midtraining_mix_0925_ingredient2_100B,
            DataMix.OLMo_longmino_mix_0625,
            DataMix.OLMo_longmino_mix_0925,
        ]:
            if tokenizer == TokenizerName.dolma2:
                tokenizer_id = "allenai/dolma3-tokenizer"
        elif tokenizer == TokenizerName.gpt_neox_olmo_dolma_v1_5:
            tokenizer_id = "gpt-neox-olmo-dolma-v1_5"

        paths = []
        labels = []
        with _get_data_mix_path(self) as mix_path:
            with mix_path.open() as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    label, path = line.split(",")
                    if "{TOKENIZER}" not in path:
                        raise ValueError(f"line {line_num + 1} in data mix '{self}' is invalid")
                    path = path.replace("{TOKENIZER}", tokenizer_id)
                    paths.append(f"{base_dir}{path}")
                    labels.append(label)
        return paths, labels


# Mix files that live under the syn_data_scaling/dolma/ subfolder.
_SYN_DATA_SCALING_DOLMA_MIXES = frozenset(
    {
        "OLMo-dolma-0.03B",
        "OLMo-dolma-0.06B",
        "OLMo-dolma-0.12B",
        "OLMo-dolma-0.15B",
        "OLMo-dolma-0.3B",
        "OLMo-dolma-0.37B",
        "OLMo-dolma-0.6B",
        "OLMo-dolma-0.74B",
        "OLMo-dolma-1.2B",
        "OLMo-dolma-1.85B",
        "OLMo-dolma-2.4B",
        "OLMo-dolma-3.7B",
        "OLMo-dolma-4.8B",
        "OLMo-dolma-7.4B",
        "OLMo-dolma-9.6B",
        "OLMo-dolma-14.8B",
        "OLMo-dolma-15.2B",
        "OLMo-dolma-19.2B",
        "OLMo-dolma-29.6B",
        "OLMo-dolma-30.4B",
        "OLMo-dolma-59.2B",
        "OLMo-dolma-60.8B",
        "OLMo-dolma-0.19B",
        "OLMo-dolma-0.38B",
        "OLMo-dolma-0.95B",
        "OLMo-dolma-1.9B",
        "OLMo-dolma-3.8B",
        "OLMo-dolma-7.6B",
        "OLMo-dolma-val",
    }
)

# Mix files that live under the syn_data_scaling/dclm/ subfolder.
_SYN_DATA_SCALING_DCLM_MIXES = frozenset(
    {
        "OLMo-dclm-chin0_05",
        "OLMo-dclm-chin0_1",
        "OLMo-dclm-chin0_25",
        "OLMo-dclm-chin0_5",
        "OLMo-dclm-chin1",
        "OLMo-dclm-chin2",
        "OLMo-dclm-chin4",
        "OLMo-dclm-chin8",
        "OLMo-dclm-chin16",
        "OLMo-dclm-para-chin0_05",
        "OLMo-dclm-para-chin0_1",
        "OLMo-dclm-para-chin0_25",
        "OLMo-dclm-para-chin0_5",
        "OLMo-dclm-para-chin1",
        "OLMo-dclm-repeat-0.3b",
        "OLMo-dclm-sample",
        "OLMo-repeat16-synthetic48",
        "OLMo-repeat32-synthetic32",
        "OLMo-repeat64-synthetic13",
        "OLMo-repeat64-synthetic32",
        "OLMo-repeat64-synthetic64",
        "OLMo-repeat64-synthetic6",
        "OLMo-synthetic-chin16",
        "OLMo-synthetic-chin16-repeat4",
        "OLMo-synthetic-chin4",
        "OLMo-synthetic-chin8",
    }
)


@contextmanager
def _get_data_mix_path(name: str) -> Generator[Path, None, None]:
    import importlib_resources

    basename = os.path.basename(str(name))
    if basename in _SYN_DATA_SCALING_DCLM_MIXES:
        rel_path = f"data/mixes/syn_data_scaling/dclm/{basename}.txt"
    elif basename in _SYN_DATA_SCALING_DOLMA_MIXES:
        rel_path = f"data/mixes/syn_data_scaling/dolma/{basename}.txt"
    else:
        rel_path = f"data/mixes/{basename}.txt"

    try:
        with importlib_resources.as_file(
            importlib_resources.files("olmo_core").joinpath(rel_path)
        ) as path:
            yield path
    finally:
        pass
