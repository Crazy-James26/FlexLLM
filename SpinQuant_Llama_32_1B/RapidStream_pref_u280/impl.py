"""Vitis implementation."""

__copyright__ = """
Copyright 2024 RapidStream Design Automation, Inc.
All Rights Reserved.
"""


from pathlib import Path

from pydantic import BaseModel, Field


class ImplConfig(BaseModel):
    """The impl config model to the tapaopt final impl phase."""

    vitis_platform: str
    port_to_clock_period: dict[str, float] = Field(
        default_factory=lambda: {"ap_clk": 3.33}
    )
    placement_strategy: str | None = Field(default=None)
    max_workers: int = Field(default=1)
    max_synth_jobs: int = Field(default=8)
    is_versal: bool = Field(default=False)

    def save_to_file(self, filename: str | Path) -> None:
        """Save the configuration to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

# Create the config object
config = ImplConfig(
    vitis_platform="xilinx_u280_gen3x16_xdma_1_202211_1",
    placement_strategy="Explore",  # or None
    max_workers=4,
    max_synth_jobs=16,
    is_versal=False
)

# Save to a file
config.save_to_file("impl_config.json")