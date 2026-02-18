import math
from .base import GPUArchitecture


class CDNA3(GPUArchitecture):
    def __init__(self, **meta):
        super().__init__(**meta)

        # Static architectural limits for CDNA3 (gfx942 class)
        self.vgpr_per_simd = 65536
        self.sgpr_per_simd = 1024
        self.lds_per_cu_bytes = 65536
        self.supports_mfma = True

        # Derive waves per SIMD dynamically
        self.max_waves_per_simd = self.max_waves_per_cu // self.simd_per_cu

    def compute_occupancy(
        self,
        vgpr_per_thread: int,
        lds_per_block_bytes: int,
        threads_per_block: int,
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
