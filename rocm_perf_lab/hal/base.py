from abc import ABC, abstractmethod


class GPUArchitecture(ABC):
    def __init__(
        self,
        arch_name: str,
        cu_count: int,
        simd_per_cu: int,
        max_waves_per_cu: int,
        wave_size: int,
        max_clock_mhz: float,
    ):
        self.arch_name = arch_name
        self.compute_units = cu_count
        self.simd_per_cu = simd_per_cu
        self.max_waves_per_cu = max_waves_per_cu
        self.wave_size = wave_size
        self.max_clock_mhz = max_clock_mhz

        # Conservative default until bandwidth is calibrated per architecture
        self.peak_bandwidth_gbps: float = 50.0

    @abstractmethod
    def compute_occupancy(
        self,
        vgpr_per_thread: int,
        lds_per_block_bytes: int,
        threads_per_block: int,
    ) -> float:
        pass

    def peak_fp32_flops(self, mode: str = "scalar") -> float:
        if mode != "scalar":
            raise NotImplementedError("Only scalar peak mode implemented")

        return (
            self.compute_units
            * self.simd_per_cu
            * self.wave_size
            * self.max_clock_mhz
            * 1e6
        )

    def theoretical_peak_bandwidth(self) -> float:
        return self.peak_bandwidth_gbps
