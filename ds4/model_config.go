package ds4

import "fmt"

// ModelConfig holds all model shape parameters — read from GGUF metadata at load time.
// Replaces the old compile-time constants to support multiple model variants.
type ModelConfig struct {
	NLayer    int // transformer layers
	NEmbd     int // embedding/hidden dimension
	NVocab    int // vocabulary size
	NHead     int // attention heads
	NHeadKV   int // KV heads (1 for MLA)
	NHeadDim  int // dimension per head (total, including NoPE + RoPE)
	NValueDim int // value dimension per head
	NRot      int // RoPE-rotated dims per head tail
	NOutGroup int // grouped output LoRA factor
	NLoraQ    int // Q low-rank bottleneck dimension
	NLoraO    int // O low-rank bottleneck dimension

	// MoE
	NExpert       int // total routed experts per layer
	NExpertUsed   int // top-K experts selected
	NExpertShared int // shared (always-on) experts
	NFFExp        int // expert FFN hidden width
	NHashLayer    int // layers with hash routing (0 if none)

	// Attention
	NSWA            int // sliding-window attention capacity
	NIndexerHead    int // indexer heads (0 if no indexer)
	NIndexerHeadDim int // indexer head dimension
	NIndexerTopK    int // indexer top-K selection

	// Hyper-connections (0 = standard residual)
	NHC             int // number of HC streams (0 or 4)
	NHCSinkhornIter int // Sinkhorn normalization iterations

	// Numeric constants
	RMSEps               float32
	HCEps                float32
	ExpertWeightScale    float32
	SwiGLUClampExp       float32
	RoPEFreqBase         float32
	RoPEScaleFactor      float32
	RoPEYarnBetaFast     float32
	RoPEYarnBetaSlow     float32
	CompressRoPEFreqBase float32
	RoPEOrigCtx          int
}

// DS4FlashConfig returns the hardcoded config for DeepSeek V4 Flash.
func DS4FlashConfig() *ModelConfig {
	return &ModelConfig{
		NLayer: NLayer, NEmbd: NEmbd, NVocab: NVocab,
		NHead: NHead, NHeadKV: NHeadKV, NHeadDim: NHeadDim, NValueDim: NValueDim,
		NRot: NRot, NOutGroup: NOutGroup, NLoraQ: NLoraQ, NLoraO: NLoraO,
		NExpert: NExpert, NExpertUsed: NExpertUsed, NExpertShared: NExpertShared,
		NFFExp: NFFExp, NHashLayer: NHashLayer,
		NSWA: NSWA, NIndexerHead: NIndexerHead, NIndexerHeadDim: NIndexerHeadDim,
		NIndexerTopK: NIndexerTopK,
		NHC:          NHC, NHCSinkhornIter: NHCSinkhornIter,
		RMSEps: RMSEps, HCEps: HCEps, ExpertWeightScale: ExpertWeightScale,
		SwiGLUClampExp: SwiGLUClampExp, RoPEFreqBase: RoPEFreqBase,
		RoPEScaleFactor: RoPEScaleFactor, RoPEYarnBetaFast: RoPEYarnBetaFast,
		RoPEYarnBetaSlow: RoPEYarnBetaSlow, CompressRoPEFreqBase: CompressRoPEFreqBase,
		RoPEOrigCtx: RoPEOrigCtx,
	}
}

// DS2LiteConfig returns the config for DeepSeek-V2-Lite (16B MoE).
func DS2LiteConfig() *ModelConfig {
	return &ModelConfig{
		NLayer: 27, NEmbd: 2048, NVocab: 102400,
		NHead: 16, NHeadKV: 1, NHeadDim: 192, NValueDim: 128,
		NRot: 64, NOutGroup: 4, NLoraQ: 512, NLoraO: 512,
		NExpert: 64, NExpertUsed: 6, NExpertShared: 2,
		NFFExp: 1408, NHashLayer: 0,
		NSWA: 128, NIndexerHead: 0, NIndexerHeadDim: 0, NIndexerTopK: 0,
		NHC: 0, NHCSinkhornIter: 0, // V2 Lite uses standard residual
		RMSEps: 1e-6, HCEps: 0, ExpertWeightScale: 1.0,
		SwiGLUClampExp: 10.0, RoPEFreqBase: 10000.0,
		RoPEScaleFactor: 1.0, RoPEYarnBetaFast: 32.0, RoPEYarnBetaSlow: 1.0,
		CompressRoPEFreqBase: 0, RoPEOrigCtx: 4096,
	}
}

// DetectModelConfig reads GGUF metadata to determine the model variant.
func DetectModelConfig(m *GGUFModel) *ModelConfig {
	arch, _ := m.MetaStr("general.architecture")

	switch arch {
	case "deepseek4":
		return DS4FlashConfig()
	case "deepseek2":
		cfg := DS2LiteConfig()
		// Override from metadata if available
		if v, ok := m.MetaU32("deepseek2.num_hidden_layers"); ok {
			cfg.NLayer = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.hidden_size"); ok {
			cfg.NEmbd = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.num_attention_heads"); ok {
			cfg.NHead = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.attention.key_length"); ok {
			cfg.NHeadDim = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.attention.value_length"); ok {
			cfg.NValueDim = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.num_experts"); ok {
			cfg.NExpert = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.num_experts_per_tok"); ok {
			cfg.NExpertUsed = int(v)
		}
		if v, ok := m.MetaU32("deepseek2.vocab_size"); ok {
			cfg.NVocab = int(v)
		}
		return cfg
	default:
		// Try to detect from tensor shapes
		fmt.Printf("[model] Unknown architecture %q, trying DS4 Flash defaults\n", arch)
		return DS4FlashConfig()
	}
}

// String returns a summary of the model config.
func (c *ModelConfig) String() string {
	hc := "none"
	if c.NHC > 0 {
		hc = fmt.Sprintf("%d streams", c.NHC)
	}
	return fmt.Sprintf("%d layers, %d dim, %d heads, %d experts (top-%d), HC=%s",
		c.NLayer, c.NEmbd, c.NHead, c.NExpert, c.NExpertUsed, hc)
}

// Cfg returns the model config from a DecodeState's engine reference.
// Falls back to DS4 Flash defaults if engine is nil.
func (ds *DecodeState) Cfg() *ModelConfig {
	if ds.Engine != nil {
		if eng, ok := ds.Engine.(*Engine); ok && eng.Config != nil {
			return eng.Config
		}
	}
	return DS4FlashConfig()
}
