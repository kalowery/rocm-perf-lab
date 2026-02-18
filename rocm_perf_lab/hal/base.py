from abc import ABC, abstractmethod


class GPUArchitecture(ABC):
    name: str
    gfx: list[str]
    wave_size: int
    compute_units: int
    simd_per_cu: int
    max_waves_per_simd: int
    vgpr_per_simd: int
    sgpr_per_simd: int
    lds_per_cu_bytes: int
    supports_mfma: bool
    peak_bandwidth_gbps: float = 50.0  # conservative default

    @abstractmethod
    def compute_occupancy(
        self,
        vgpr_per_thread: int,
        lds_per_block_bytes: int,
        threads_per_block: int
    ) -> float:
        pass

    @abstractmethod
    def theoretical_peak_flops(self, clock_mhz: float) -> float:
        pass

    def theoretical_peak_bandwidth(self) -> float:
        return self.peak_bandwidth_gbps
