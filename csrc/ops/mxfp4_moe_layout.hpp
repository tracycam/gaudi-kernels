#pragma once
namespace gaudi_kernels::mxfp4_moe_layout {
// Shape alone cannot identify these lossless but incompatible byte layouts.
enum class WeightLayout : int { Unspecified=0, HistoricalN512=1, NativeN512V2=2 };
inline bool supported(WeightLayout layout){return layout==WeightLayout::HistoricalN512||layout==WeightLayout::NativeN512V2;}
}
