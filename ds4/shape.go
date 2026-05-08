package ds4

// Model shape constants — fixed for DeepSeek V4 Flash.
// The loader validates GGUF metadata against these exact values.
const (
	NLayer          = 43
	NEmbd           = 4096
	NVocab          = 129280
	NHead           = 64
	NHeadKV         = 1
	NHeadDim        = 512
	NValueDim       = 512
	NRot            = 64   // RoPE-rotated dims per head tail
	NOutGroup       = 8    // grouped output LoRA
	NLoraQ          = 1024 // Q low-rank bottleneck
	NLoraO          = 1024 // O low-rank bottleneck
	NExpert         = 256
	NExpertUsed     = 6
	NExpertUsedFast = 4 // reduced expert count for speed mode
	NExpertShared   = 1
	NFFExp          = 2048 // expert FFN hidden width
	NHashLayer      = 3    // layers with token-ID→expert-ID hash routing
	NSWA            = 128  // sliding-window attention capacity
	NIndexerHead    = 64
	NIndexerHeadDim = 128
	NIndexerTopK    = 512
	NHC             = 4 // hyper-connection streams
	NHCSinkhornIter = 20

	RMSEps               = 1e-6
	HCEps                = 1e-6
	ExpertWeightScale    = 1.5
	SwiGLUClampExp       = 10.0
	RoPEFreqBase         = 10000.0
	RoPEScaleFactor      = 16.0
	RoPEYarnBetaFast     = 32.0
	RoPEYarnBetaSlow     = 1.0
	CompressRoPEFreqBase = 160000.0
	RoPEOrigCtx          = 65536

	// Quantization block sizes
	QK_K = 256 // superblock size for K-quants
)

// GGUF tensor type IDs
const (
	TensorF32    = 0
	TensorF16    = 1
	TensorQ8_0   = 8
	TensorQ2_K   = 10
	TensorQ4_K   = 12
	TensorIQ2XXS = 16
	TensorI32    = 26
)

// Block sizes in bytes per QK_K=256 elements
const (
	BlockQ2KSize    = 84  // 16 scales + 64 qs + 2 d + 2 dmin
	BlockQ4KSize    = 144 // 2 d + 2 dmin + 12 scales + 128 qs
	BlockIQ2XXSSize = 66  // 2 d + 64 qs (codebook indices)
	BlockQ8KSize    = 292 // 4 d + 256 qs + 32 bsums
	BlockQ8_0Size   = 34  // 2 d(F16) + 32 qs (per 32 elements)
)

// TensorTypeSize returns bytes per element (or per-block info) for a GGUF type.
func TensorTypeSize(typ uint32) (bytesPerBlock int, elementsPerBlock int) {
	switch typ {
	case TensorF32:
		return 4, 1
	case TensorF16:
		return 2, 1
	case TensorQ8_0:
		return BlockQ8_0Size, 32
	case TensorQ2_K:
		return BlockQ2KSize, QK_K
	case TensorQ4_K:
		return BlockQ4KSize, QK_K
	case TensorIQ2XXS:
		return BlockIQ2XXSSize, QK_K
	case TensorI32:
		return 4, 1
	default:
		return 0, 0
	}
}
