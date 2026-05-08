package gpu

import _ "embed"

//go:embed shaders/gemv_f32.spv
var SPIRVGemvF32 []byte

//go:embed shaders/gemv_q8_0_f16scale.spv
var SPIRVGemvQ8_0F16Scale []byte
