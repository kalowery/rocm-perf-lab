from typing import Optional, Tuple
from pydantic import BaseModel


class KernelModel(BaseModel):
    name: Optional[str]
    grid: Optional[Tuple[int, int, int]]
    block: Optional[Tuple[int, int, int]]


class GPUModel(BaseModel):
    architecture: str
    wave_size: int
    compute_units: int
    theoretical_peak_flops: Optional[float] = None


class StabilityModel(BaseModel):
    runs: int
    mean_ms: float
    stddev_ms: float
    cv: float
    classification: str


class ResourcesModel(BaseModel):
    vgpr_per_thread: Optional[int]
    sgpr_per_wave: Optional[int]
    lds_bytes: Optional[int]


class OccupancyModel(BaseModel):
    theoretical: float
    threads_per_block: int
    wave_size: int


class RooflineModel(BaseModel):
    flops: float
    bytes: float
    arithmetic_intensity: float
    achieved_gflops: float
    achieved_bandwidth_gbps: float
    bound: str


class ProfileModel(BaseModel):
    schema_version: str
    kernel: KernelModel
    gpu: GPUModel
    runtime_ms: float
    stability: StabilityModel
    resources: Optional[ResourcesModel]
    occupancy: Optional[OccupancyModel]
    roofline: Optional[RooflineModel]
