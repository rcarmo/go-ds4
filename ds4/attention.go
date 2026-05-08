package ds4

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// DecodeState holds pre-allocated buffers for single-token decode.
type DecodeState struct {
	// HC state
	CurHC  []float32 // [NHC * NEmbd] current HC state
	NextHC []float32 // [NHC * NEmbd] next HC state

	// Attention
	AttnCur    []float32 // [NEmbd] HC-pre attention input
	AttnNormed []float32 // [NEmbd] normed attention input
	QR         []float32 // [NLoraQ] Q LoRA bottleneck
	QRNorm     []float32 // [NLoraQ] Q LoRA normed
	Q          []float32 // [NHead * NHeadDim] full Q
	KV         []float32 // [NHeadDim] compressed KV
	Heads      []float32 // [NHead * NHeadDim] attention output
	AttnOut    []float32 // [NEmbd] after output projection
	AttnScore  []float32 // attention scores (sized per layer)
	KVCacheRow []float32 // [NHeadDim] scratch row for cache quantization
	TmpLoRA    []float32 // [NLoraO] scratch for output grouped LoRA

	// Compressor/indexer scratch (decode)
	CompKVCur       []float32 // [2*NHeadDim]
	CompScoreCur    []float32 // [2*NHeadDim]
	CompPooled      []float32 // [NHeadDim]
	CompOut         []float32 // [NHeadDim]
	IndexCompKVCur  []float32 // [2*NIndexerHeadDim]
	IndexCompScore  []float32 // [2*NIndexerHeadDim]
	IndexCompPooled []float32 // [NIndexerHeadDim]
	IndexCompOut    []float32 // [NIndexerHeadDim]
	IndexQ          []float32 // [NIndexerHead*NIndexerHeadDim]
	IndexWeights    []float32 // [NIndexerHead]
	IndexScores     []float32 // [ctx/4+2]
	IndexAllowed    []bool    // [ctx/4+2]

	// FFN / MoE
	FfnCur      []float32     // [NEmbd] HC-pre FFN input
	FfnNormed   []float32     // [NEmbd] normed FFN input
	RoutedXQ    []byte        // Q8_K quantized activation for experts
	RoutedMidQ  []byte        // Q8_K quantized expert hidden
	RoutedOut   []float32     // [NEmbd] routed expert output
	SharedOut   []float32     // [NEmbd] shared expert output
	SharedGate  []float32     // [NFFExp] shared expert gate scratch
	SharedUp    []float32     // [NFFExp] shared expert up scratch
	RouteLogits []float32     // [NExpert] routing logits scratch
	RouteScores []expertScore // [NExpert] routing score scratch
	ExpertMidQ  []byte        // [NExpertUsed, (NFFExp/QK_K)*BlockQ8KSize]
	ExpertGate  []float32     // [NExpertUsed, NFFExp]
	ExpertUp    []float32     // [NExpertUsed, NFFExp]
	ExpertOut   []float32     // [NExpertUsed, NEmbd]

	// HC/output scratch
	AttnResidual []float32 // [NHC*NEmbd]
	FfnResidual  []float32 // [NHC*NEmbd]
	OutFlat      []float32 // [NHC*NEmbd]
	OutHCWeights []float32 // [NHC]
	OutCollapsed []float32 // [NEmbd]
	HCFlat       []float32 // [NHC*NEmbd] HC pre scratch
	HCMix        []float32 // [2*NHC+NHC^2] HC pre scratch
	HCSumTmp     []float32 // [NEmbd] HC post sum scratch
	Post         []float32 // [NHC]
	Comb         []float32 // [NHC * NHC]

	// Engine back-reference (for GPU dispatch)
	Engine   interface{}
	LayerIdx int // current layer index (set during forward pass)
}

// NewDecodeState allocates decode buffers for a given context size.
func NewDecodeState(ctxSize int) *DecodeState {
	maxScores := NSWA + ctxSize/2 + 2 // raw + max compressed
	return &DecodeState{
		CurHC:           make([]float32, hcDim),
		NextHC:          make([]float32, hcDim),
		AttnCur:         make([]float32, NEmbd),
		AttnNormed:      make([]float32, NEmbd),
		QR:              make([]float32, NLoraQ),
		QRNorm:          make([]float32, NLoraQ),
		Q:               make([]float32, NHead*NHeadDim),
		KV:              make([]float32, NHeadDim),
		Heads:           make([]float32, NHead*NHeadDim),
		AttnOut:         make([]float32, NEmbd),
		AttnScore:       make([]float32, maxScores),
		KVCacheRow:      make([]float32, NHeadDim),
		TmpLoRA:         make([]float32, NLoraO),
		CompKVCur:       make([]float32, 2*NHeadDim),
		CompScoreCur:    make([]float32, 2*NHeadDim),
		CompPooled:      make([]float32, NHeadDim),
		CompOut:         make([]float32, NHeadDim),
		IndexCompKVCur:  make([]float32, 2*NIndexerHeadDim),
		IndexCompScore:  make([]float32, 2*NIndexerHeadDim),
		IndexCompPooled: make([]float32, NIndexerHeadDim),
		IndexCompOut:    make([]float32, NIndexerHeadDim),
		IndexQ:          make([]float32, NIndexerHead*NIndexerHeadDim),
		IndexWeights:    make([]float32, NIndexerHead),
		IndexScores:     make([]float32, ctxSize/4+2),
		IndexAllowed:    make([]bool, ctxSize/4+2),
		FfnCur:          make([]float32, NEmbd),
		FfnNormed:       make([]float32, NEmbd),
		RoutedXQ:        make([]byte, (NEmbd/QK_K)*BlockQ8KSize),
		RoutedMidQ:      make([]byte, NExpertUsed*(NFFExp/QK_K)*BlockQ8KSize),
		RoutedOut:       make([]float32, NEmbd),
		SharedOut:       make([]float32, NEmbd),
		SharedGate:      make([]float32, NFFExp),
		SharedUp:        make([]float32, NFFExp),
		RouteLogits:     make([]float32, NExpert),
		RouteScores:     make([]expertScore, NExpert),
		ExpertMidQ:      make([]byte, NExpertUsed*(NFFExp/QK_K)*BlockQ8KSize),
		ExpertGate:      make([]float32, NExpertUsed*NFFExp),
		ExpertUp:        make([]float32, NExpertUsed*NFFExp),
		ExpertOut:       make([]float32, NExpertUsed*NEmbd),
		AttnResidual:    make([]float32, hcDim),
		FfnResidual:     make([]float32, hcDim),
		OutFlat:         make([]float32, hcDim),
		OutHCWeights:    make([]float32, NHC),
		OutCollapsed:    make([]float32, NEmbd),
		HCFlat:          make([]float32, hcDim),
		HCMix:           make([]float32, hcMixDim),
		HCSumTmp:        make([]float32, NEmbd),
		Post:            make([]float32, NHC),
		Comb:            make([]float32, NHC*NHC),
	}
}

// layerAttnDecode runs MLA attention for a single decode token.
func layerAttnDecode(
	ds *DecodeState,
	layer *LayerWeights,
	cache *LayerCache,
	model *GGUFModel,
	pos, il int,
) {
	// 1. RMSNorm attention input
	normW := tensorF32Unsafe(layer.AttnNorm)
	copy(ds.AttnNormed, ds.AttnCur)
	rmsNorm(ds.AttnNormed, normW)

	cfg := ds.Cfg()
	isV2 := len(layer.AttnQ) > 0 // V2 uses direct AttnQ, V4 uses AttnQA+AttnQB

	if isV2 {
		// V2 Lite: direct Q projection + split KV
		// Q: [NEmbd, NHead*NHeadDim]
		matvecAuto(ds.Q, layer.AttnQ, ds.AttnNormed, cfg.NEmbd, cfg.NHead*(cfg.NHeadDim))

		// KV: two-stage: kv_a_mqa [NEmbd, kv_lora_rank+NRot] then kv_b [kv_lora_rank, NHead*NHeadDim]
		// For simplicity: compute KV_A, norm, then KV_B
		kvLoraRank := cfg.NLoraQ // reuse lora rank field
		if kvLoraRank == 0 {
			kvLoraRank = 512
		}
		kvA := make([]float32, kvLoraRank+cfg.NRot)
		matvecAuto(kvA, layer.AttnKVA_MQA, ds.AttnNormed, cfg.NEmbd, kvLoraRank+cfg.NRot)

		// Norm the non-RoPE part
		if len(layer.AttnKVANorm) > 0 {
			kvNormW := tensorF32Unsafe(layer.AttnKVANorm)
			rmsNorm(kvA[:kvLoraRank], kvNormW)
		}

		// KV_B: expand [kvLoraRank] -> [NHead*NHeadDim]
		matvecAuto(ds.KV, layer.AttnKVB, kvA[:kvLoraRank], kvLoraRank, cfg.NHeadDim)

		// Per-head RMSNorm on Q
		for h := 0; h < cfg.NHead; h++ {
			head := ds.Q[h*cfg.NHeadDim : (h+1)*cfg.NHeadDim]
			rmsNormNoScale(head)
		}
	} else {
		// V4 Flash: LoRA Q projection
		fusedOK := false
		if ds.Engine != nil {
			if eng, ok := ds.Engine.(*Engine); ok {
				fusedOK = eng.gpuFusedAttnQAKV(ds.QR, ds.KV, ds.AttnNormed, il)
			}
		}
		if !fusedOK {
			matvecQ8_0GPULayer(ds.QR, layer.AttnQA, ds.AttnNormed, NEmbd, NLoraQ, ds, "attn_q_a.weight")
			matvecQ8_0GPULayer(ds.KV, layer.AttnKV, ds.AttnNormed, NEmbd, NHeadDim, ds, "attn_kv.weight")
		}
		qrNormW := tensorF32Unsafe(layer.AttnQANorm)
		copy(ds.QRNorm, ds.QR)
		rmsNorm(ds.QRNorm, qrNormW)
		matvecQ8_0GPULayer(ds.Q, layer.AttnQB, ds.QRNorm, NLoraQ, NHead*NHeadDim, ds, "attn_q_b.weight")
		for h := 0; h < cfg.NHead; h++ {
			rmsNormNoScale(ds.Q[h*cfg.NHeadDim : (h+1)*cfg.NHeadDim])
		}

		// KV norm
		kvNormW := tensorF32Unsafe(layer.AttnKVANorm)
		rmsNorm(ds.KV, kvNormW)
	}

	// Common: RoPE on Q and KV
	freqBase := layerRoPEFreqBase(il)
	freqScale := layerRoPEFreqScale(il)
	ropeYaRNTailInplace(ds.Q, pos, cfg.NHead, cfg.NHeadDim, cfg.NRot, freqBase, freqScale, false)

	// Apply RoPE to KV tail
	ropeYaRNTailInplace(ds.KV, pos, 1, cfg.NHeadDim, NRot, freqBase, freqScale, false)

	// FP8 quantize the non-RoPE portion for storage (reuse scratch to avoid alloc)
	copy(ds.KVCacheRow, ds.KV)
	fp8KVQuantizeInplace(ds.KVCacheRow, cfg.NHeadDim, NRot)

	// Push to cache
	cache.PushRawKV(ds.KVCacheRow)

	// 4. Compressor/indexer update
	ratio := cache.CompressRatio
	var compAllowed []bool
	if ratio > 0 && len(layer.CompressorKV) != 0 {
		if compressorDecodeOne(ds.CompOut, ds.CompKVCur, ds.CompScoreCur, ds.CompPooled,
			layer.CompressorKV, layer.CompressorGate, layer.CompressorAPE, layer.CompressorNorm,
			"attn_compressor_kv.weight", "attn_compressor_gate.weight",
			ds.AttnNormed, cache.CompStateKV, cache.CompStateScore,
			NHeadDim, ratio, il, pos, true, ds) {
			cache.PushCompKV(ds.CompOut)
		}
		if ratio == 4 && len(layer.IndexerCompKV) != 0 {
			if compressorDecodeOne(ds.IndexCompOut, ds.IndexCompKVCur, ds.IndexCompScore, ds.IndexCompPooled,
				layer.IndexerCompKV, layer.IndexerCompGate, layer.IndexerCompAPE, layer.IndexerCompNorm,
				"indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
				ds.AttnNormed, cache.IndexStateKV, cache.IndexStateScore,
				NIndexerHeadDim, ratio, il, pos, false, ds) {
				cache.PushIndexCompKV(ds.IndexCompOut)
			}
			compAllowed = indexerAllowedDecodeOne(ds, layer, ds.AttnCur, ds.QRNorm, cache, il, pos)
		}
	}

	// 5. Attention scoring over raw SWA + optional compressed rows
	nRaw := cache.NRaw
	if nRaw > cache.CapRaw {
		nRaw = cache.CapRaw
	}
	nComp := cache.NComp
	if nComp > cache.CompCap {
		nComp = cache.CompCap
	}
	nTotal := nRaw + nComp

	// Try GPU batched attention (raw-only for now, no compressed)
	gpuAttnOK := false
	if nComp == 0 && ds.Engine != nil {
		if eng, ok := ds.Engine.(*Engine); ok {
			gpuAttnOK = eng.gpuAttnScoring(ds, cache, layer, il)
		}
	}

	if !gpuAttnOK {
		// CPU fallback
		rawStart := cache.rawStart()
		compStart := cache.compStart()
		var sinks []float32
		if len(layer.AttnSinks) > 0 {
			sinks = tensorF32Unsafe(layer.AttnSinks)
		} else {
			sinks = make([]float32, NHead) // zero sinks for V2
		}
		scale := float32(1.0 / math.Sqrt(float64(cfg.NHeadDim)))

		for h := 0; h < cfg.NHead; h++ {
			qHead := ds.Q[h*cfg.NHeadDim : (h+1)*cfg.NHeadDim]
			maxScore := sinks[h]

			for t := 0; t < nRaw; t++ {
				idx := rawStart + t
				if idx >= cache.CapRaw {
					idx -= cache.CapRaw
				}
				kvRow := cache.RawKV[idx*cfg.NHeadDim : (idx+1)*cfg.NHeadDim]
				s := simd.Sdot(qHead, kvRow) * scale
				ds.AttnScore[t] = s
				if s > maxScore {
					maxScore = s
				}
			}
			for t := 0; t < nComp; t++ {
				if compAllowed != nil && !compAllowed[t] {
					ds.AttnScore[nRaw+t] = -1e30
					continue
				}
				idx := compStart + t
				if idx >= cache.CompCap {
					idx -= cache.CompCap
				}
				kvRow := cache.CompKV[idx*cfg.NHeadDim : (idx+1)*cfg.NHeadDim]
				s := simd.Sdot(qHead, kvRow) * scale
				ds.AttnScore[nRaw+t] = s
				if s > maxScore {
					maxScore = s
				}
			}

			headOut := ds.Heads[h*cfg.NHeadDim : (h+1)*cfg.NHeadDim]
			for d := 0; d < cfg.NHeadDim; d++ {
				headOut[d] = 0
			}
			denom := float32(math.Exp(float64(sinks[h] - maxScore)))

			for t := 0; t < nRaw; t++ {
				w := float32(math.Exp(float64(ds.AttnScore[t] - maxScore)))
				denom += w
				idx := rawStart + t
				if idx >= cache.CapRaw {
					idx -= cache.CapRaw
				}
				kvRow := cache.RawKV[idx*cfg.NHeadDim : (idx+1)*cfg.NHeadDim]
				simd.Saxpy(w, kvRow, headOut)
			}
			for t := 0; t < nComp; t++ {
				if ds.AttnScore[nRaw+t] <= -1e29 {
					continue
				}
				w := float32(math.Exp(float64(ds.AttnScore[nRaw+t] - maxScore)))
				denom += w
				idx := compStart + t
				if idx >= cache.CompCap {
					idx -= cache.CompCap
				}
				kvRow := cache.CompKV[idx*cfg.NHeadDim : (idx+1)*cfg.NHeadDim]
				simd.Saxpy(w, kvRow, headOut)
			}

			if denom > 0 {
				inv := 1 / denom
				for d := 0; d < cfg.NHeadDim; d++ {
					headOut[d] *= inv
				}
			}
			_ = nTotal
		}
	} // end if !gpuAttnOK

	// 5. Output projection
	ropeYaRNTailInplace(ds.Heads, pos, cfg.NHead, cfg.NHeadDim, cfg.NRot, freqBase, freqScale, true)

	if isV2 {
		// V2: direct output projection (may be Q3_K or Q8_0)
		matvecAuto(ds.AttnOut, layer.AttnOutput, ds.Heads, cfg.NHead*cfg.NHeadDim, cfg.NEmbd)
	} else {
		// V4: grouped LoRA output
		matvecQ8_0Grouped(ds.TmpLoRA, layer.AttnOutputA, ds.Heads, NHead*NValueDim, NLoraO, NOutGroup)
		matvecQ8_0GPULayer(ds.AttnOut, layer.AttnOutputB, ds.TmpLoRA, NLoraO, NEmbd, ds, "attn_output_b.weight")
	}
}

// layerRoPEFreqBase returns the RoPE frequency base for this layer.
func layerRoPEFreqBase(il int) float32 {
	if layerCompressRatio(il) > 0 {
		return CompressRoPEFreqBase
	}
	return RoPEFreqBase
}

// layerRoPEFreqScale returns the RoPE frequency scale for this layer.
func layerRoPEFreqScale(il int) float32 {
	if layerCompressRatio(il) > 0 {
		return 1.0 / RoPEScaleFactor
	}
	return 1.0
}

// tensorF32Unsafe returns a float32 slice from raw bytes (no bounds check).
func tensorF32Unsafe(data []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
}
