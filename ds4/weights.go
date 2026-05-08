package ds4

import "fmt"

// LayerWeights holds tensor references for one transformer layer.
// All fields are byte slices pointing into the mmap (zero-copy).
type LayerWeights struct {
	// Hyper-connection (attention sublayer)
	HCAttnFn    []byte // F16 [NHC*NEmbd, 2*NHC+NHC²]
	HCAttnScale []byte // F32 [3]
	HCAttnBase  []byte // F32 [2*NHC+NHC²]
	AttnNorm    []byte // F32 [NEmbd]

	// MLA: Q projection (low-rank LoRA)
	AttnQA     []byte // Q8_0 [NEmbd, NLoraQ]
	AttnQANorm []byte // F32  [NLoraQ]
	AttnQB     []byte // Q8_0 [NLoraQ, NHead*NHeadDim]

	// MLA: KV projection
	AttnKV      []byte // Q8_0 [NEmbd, NHeadDim]
	AttnKVANorm []byte // F32  [NHeadDim]
	AttnSinks   []byte // attention sink vectors

	// MLA: output projection (grouped LoRA for V4, direct for V2)
	AttnOutputA []byte // Q8_0 [NHead*NValueDim, NLoraO] (V4 only)
	AttnOutputB []byte // Q8_0 [NLoraO, NEmbd] grouped (V4 only)
	AttnOutput  []byte // direct output projection (V2 only)

	// V2 Lite: split KV projection
	AttnQ       []byte // direct Q projection (V2 only, replaces AttnQA+AttnQB)
	AttnKVA_MQA []byte // KV projection part A (V2 only)
	AttnKVB     []byte // KV projection part B (V2 only)

	// Compressor (compressed-KV layers only, nil for others)
	CompressorAPE  []byte
	CompressorKV   []byte
	CompressorGate []byte
	CompressorNorm []byte

	// Indexer (ratio-4 layers only, nil for others)
	IndexerQB       []byte
	IndexerProj     []byte
	IndexerCompAPE  []byte
	IndexerCompKV   []byte
	IndexerCompGate []byte
	IndexerCompNorm []byte

	// Hyper-connection (FFN sublayer)
	HCFfnFn    []byte // F16 [NHC*NEmbd, 2*NHC+NHC²]
	HCFfnScale []byte // F32 [3]
	HCFfnBase  []byte // F32 [2*NHC+NHC²]
	FfnNorm    []byte // F32 [NEmbd]

	// MoE routing
	FfnGateTid2Eid []byte // I32 [NExpertUsed, NVocab] (hash layers only)
	FfnGateInp     []byte // F16 [NEmbd, NExpert]
	FfnExpProbsB   []byte // F32 [NExpert]

	// Routed experts (stored as contiguous blocks for all experts)
	FfnGateExps []byte // IQ2_XXS [NFFExp, NExpert*NEmbd]
	FfnUpExps   []byte // IQ2_XXS [NFFExp, NExpert*NEmbd]
	FfnDownExps []byte // Q2_K    [NEmbd, NExpert*NFFExp]

	// Shared expert
	FfnGateShexp []byte // Q8_0 [NFFExp, NEmbd]
	FfnUpShexp   []byte // Q8_0 [NFFExp, NEmbd]
	FfnDownShexp []byte // Q8_0 [NEmbd, NFFExp]
}

// Weights holds all model weights with tensor data pointing into the mmap.
type Weights struct {
	TokenEmbd     []byte // F16  [NEmbd, NVocab]
	OutputHCBase  []byte // F32  [NHC]
	OutputHCFn    []byte // F16  [NHC*NEmbd, NHC]
	OutputHCScale []byte // F32  [1]
	OutputNorm    []byte // F32  [NEmbd]
	Output        []byte // Q8_0 [NEmbd, NVocab]
	Layer         [NLayer]LayerWeights
}

// BindWeights resolves tensor names from the GGUF to weight struct fields.
func BindWeights(m *GGUFModel) (*Weights, error) {
	w := &Weights{}
	var err error

	bind := func(dst *[]byte, name string) {
		if err != nil {
			return
		}
		d, e := m.TensorData(name)
		if e != nil {
			err = fmt.Errorf("bind %s: %w", name, e)
			return
		}
		*dst = d
	}

	bindOpt := func(dst *[]byte, name string) {
		if err != nil {
			return
		}
		d, e := m.TensorData(name)
		if e == nil {
			*dst = d
		}
		// optional: nil is fine
	}

	// Global tensors
	bind(&w.TokenEmbd, "token_embd.weight")
	bind(&w.OutputHCBase, "output_hc_base.weight")
	bind(&w.OutputHCFn, "output_hc_fn.weight")
	bind(&w.OutputHCScale, "output_hc_scale.weight")
	bind(&w.OutputNorm, "output_norm.weight")
	bindOpt(&w.Output, "output.weight")

	// Per-layer tensors
	for il := 0; il < NLayer; il++ { // TODO: use config.NLayer
		l := &w.Layer[il]
		p := fmt.Sprintf("blk.%d.", il)

		bind(&l.HCAttnFn, p+"hc_attn_fn.weight")
		bind(&l.HCAttnScale, p+"hc_attn_scale.weight")
		bind(&l.HCAttnBase, p+"hc_attn_base.weight")
		bind(&l.AttnNorm, p+"attn_norm.weight")

		bind(&l.AttnQA, p+"attn_q_a.weight")
		bind(&l.AttnQANorm, p+"attn_q_a_norm.weight")
		bind(&l.AttnQB, p+"attn_q_b.weight")

		bind(&l.AttnKV, p+"attn_kv.weight")
		bind(&l.AttnKVANorm, p+"attn_kv_a_norm.weight")
		bindOpt(&l.AttnSinks, p+"attn_sinks.weight")

		bind(&l.AttnOutputA, p+"attn_output_a.weight")
		bind(&l.AttnOutputB, p+"attn_output_b.weight")

		// Compressor (optional, only compressed-KV layers)
		bindOpt(&l.CompressorAPE, p+"attn_compressor_ape.weight")
		bindOpt(&l.CompressorKV, p+"attn_compressor_kv.weight")
		bindOpt(&l.CompressorGate, p+"attn_compressor_gate.weight")
		bindOpt(&l.CompressorNorm, p+"attn_compressor_norm.weight")

		// Indexer (optional, only ratio-4 layers)
		bindOpt(&l.IndexerQB, p+"indexer_attn_q_b.weight")
		bindOpt(&l.IndexerProj, p+"indexer_proj.weight")
		bindOpt(&l.IndexerCompAPE, p+"indexer_compressor_ape.weight")
		bindOpt(&l.IndexerCompKV, p+"indexer_compressor_kv.weight")
		bindOpt(&l.IndexerCompGate, p+"indexer_compressor_gate.weight")
		bindOpt(&l.IndexerCompNorm, p+"indexer_compressor_norm.weight")

		bind(&l.HCFfnFn, p+"hc_ffn_fn.weight")
		bind(&l.HCFfnScale, p+"hc_ffn_scale.weight")
		bind(&l.HCFfnBase, p+"hc_ffn_base.weight")
		bind(&l.FfnNorm, p+"ffn_norm.weight")

		// Routing: hash table (optional, only hash layers)
		bindOpt(&l.FfnGateTid2Eid, p+"ffn_gate_tid2eid.weight")
		bind(&l.FfnGateInp, p+"ffn_gate_inp.weight")
		bindOpt(&l.FfnExpProbsB, p+"exp_probs_b.bias")

		// Experts
		bind(&l.FfnGateExps, p+"ffn_gate_exps.weight")
		bind(&l.FfnUpExps, p+"ffn_up_exps.weight")
		bind(&l.FfnDownExps, p+"ffn_down_exps.weight")

		// Shared expert
		bind(&l.FfnGateShexp, p+"ffn_gate_shexp.weight")
		bind(&l.FfnUpShexp, p+"ffn_up_shexp.weight")
		bind(&l.FfnDownShexp, p+"ffn_down_shexp.weight")
	}

	if err != nil {
		return nil, err
	}
	return w, nil
}

// BindWeightsV2 resolves tensor names for DeepSeek-V2-Lite GGUF layout.
func BindWeightsV2(m *GGUFModel, nLayers int) (*Weights, error) {
	w := &Weights{}
	var err error

	bind := func(dst *[]byte, name string) {
		if err != nil {
			return
		}
		d, e := m.TensorData(name)
		if e != nil {
			err = fmt.Errorf("bind %s: %w", name, e)
			return
		}
		*dst = d
	}
	bindOpt := func(dst *[]byte, name string) {
		if err != nil {
			return
		}
		d, e := m.TensorData(name)
		if e == nil {
			*dst = d
		}
	}

	bind(&w.TokenEmbd, "token_embd.weight")
	bind(&w.OutputNorm, "output_norm.weight")
	bindOpt(&w.Output, "output.weight")
	// V2 has no HC output tensors — OutputHCBase/Fn/Scale stay nil

	for il := 0; il < nLayers; il++ {
		l := &w.Layer[il]
		p := fmt.Sprintf("blk.%d.", il)

		bind(&l.AttnNorm, p+"attn_norm.weight")

		// V2 attention: direct Q (not LoRA), split KV, direct output
		bind(&l.AttnQ, p+"attn_q.weight")
		bind(&l.AttnKVA_MQA, p+"attn_kv_a_mqa.weight")
		bindOpt(&l.AttnKVANorm, p+"attn_kv_a_norm.weight")
		bind(&l.AttnKVB, p+"attn_kv_b.weight")
		bind(&l.AttnOutput, p+"attn_output.weight")

		// FFN norm
		bind(&l.FfnNorm, p+"ffn_norm.weight")

		// Routing (optional — layer 0 is dense in V2 Lite)
		bindOpt(&l.FfnGateInp, p+"ffn_gate_inp.weight")
		bindOpt(&l.FfnExpProbsB, p+"exp_probs_b.bias")

		// Experts (optional — layer 0 has no experts)
		bindOpt(&l.FfnGateExps, p+"ffn_gate_exps.weight")
		bindOpt(&l.FfnUpExps, p+"ffn_up_exps.weight")
		bindOpt(&l.FfnDownExps, p+"ffn_down_exps.weight")

		// Shared expert
		bindOpt(&l.FfnGateShexp, p+"ffn_gate_shexp.weight")
		bindOpt(&l.FfnUpShexp, p+"ffn_up_shexp.weight")
		bindOpt(&l.FfnDownShexp, p+"ffn_down_shexp.weight")

		// Dense FFN (layer 0 in V2 Lite — uses ffn_gate/up/down without _shexp suffix)
		// Only bind if expert weights not found (to avoid overwriting shared expert)
		if l.FfnGateExps == nil {
			bindOpt(&l.FfnGateShexp, p+"ffn_gate.weight")
			bindOpt(&l.FfnUpShexp, p+"ffn_up.weight")
			bindOpt(&l.FfnDownShexp, p+"ffn_down.weight")
		}

	}

	if err != nil {
		return nil, err
	}
	return w, nil
}
