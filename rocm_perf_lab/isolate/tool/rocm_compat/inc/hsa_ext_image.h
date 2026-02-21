#pragma once

// Compatibility shim for ROCm header layout mismatch.
// hsa_api_trace.h includes "inc/hsa_ext_image.h"
// but some ROCm installations place hsa_ext_image.h directly under hsa/.

#include <hsa/hsa_ext_image.h>
