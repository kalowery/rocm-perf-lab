import math
from .base import GPUArchitecture


class RDNA2(GPUArchitecture):
    def __init__(self, compute_units: int):
        self.name = "rdna2"
        self.gfx = ["gfx1030", "gfx1031", "gfx1032"]
        self.wave_size = 32
        self.compute_units = compute_units
        self.simd_per_cu = 4
        self.max_waves_per_simd = 8
        self.vgpr_per_simd = 16384
        self.sgpr_per_simd = 800
        self.lds_per_cu_bytes = 65536
        self.supports_mfma = False
        self.peak_bandwidth_gbps = 50.0  # conservative APU default

    def compute_occupancy(
        self,
        vgpr_per_thread: int,
        lds_per_block_bytes: int,
        threads_per_block: int
    ) -> float:
        wave_size = self.wave_size
        waves_per_block = math.ceil(threads_per_block / wave_size)

        vgpr_wave = vgpr_per_thread * wave_size
        if vgpr_wave == 0:
            return 0.0

        w_vgpr = self.vgpr_per_simd // vgpr_wave
        w_hw = self.max_waves_per_simd

        if lds_per_block_bytes > 0:
            blocks_per_cu = self.lds_per_cu_bytes // lds_per_block_bytes
            waves_per_cu = blocks_per_cu * waves_per_block
            w_lds = waves_per_cu // self.simd_per_cu
        else:
            w_lds = w_hw

        w_active = min(w_vgpr, w_hw, w_lds)
        return max(0.0, min(1.0, w_active / w_hw))

    def theoretical_peak_flops(self, clock_mhz: float) -> float:
        alus_per_simd = 32
        flops_per_cycle = alus_per_simd * 2
        total_simd = self.compute_units * self.simd_per_cu
        clock_hz = clock_mhz * 1e6
        return total_simd * flops_per_cycle * clock_hz
