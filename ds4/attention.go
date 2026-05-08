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
	AttnNormed []float32 // [NEmbd] normed input
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
	FfnNormed  []float32 // [NEmbd]
	RoutedXQ   []byte    // Q8_K quantized activation for experts
	RoutedMidQ []byte    // Q8_K quantized expert hidden
	RoutedOut  []float32 // [NEmbd] routed expert output
	SharedOut  []float32 // [NEmbd] shared expert output
	GateUp     []float32 // [NFFExp] gate/up intermediate

	// HC scratch
	Post []float32 // [NHC]
	Comb []float32 // [NHC * NHC]
}

// NewDecodeState allocates decode buffers for a given context size.
func NewDecodeState(ctxSize int) *DecodeState {
	maxScores := NSWA + ctxSize/2 + 2 // raw + max compressed
	return &DecodeState{
		CurHC:      make([]float32, hcDim),
		NextHC:     make([]float32, hcDim),
		AttnNormed: make([]float32, NEmbd),
		QR:         make([]float32, NLoraQ),
		QRNorm:     make([]float32, NLoraQ),
		Q:          make([]float32, NHead*NHeadDim),
		KV:         make([]float32, NHeadDim),
		Heads:      make([]float32, NHead*NHeadDim),
		AttnOut:    make([]float32, NEmbd),
		AttnScore:  make([]float32, maxScores),
		KVCacheRow: make([]float32, NHeadDim),
		TmpLoRA:    make([]float32, NLoraO),
		CompKVCur:  make([]float32, 2*NHeadDim),
		CompScoreCur: make([]float32, 2*NHeadDim),
		CompPooled: make([]float32, NHeadDim),
		CompOut:    make([]float32, NHeadDim),
		IndexCompKVCur:  make([]float32, 2*NIndexerHeadDim),
		IndexCompScore:  make([]float32, 2*NIndexerHeadDim),
		IndexCompPooled: make([]float32, NIndexerHeadDim),
		IndexCompOut:    make([]float32, NIndexerHeadDim),
		IndexQ:          make([]float32, NIndexerHead*NIndexerHeadDim),
		IndexWeights:    make([]float32, NIndexerHead),
		IndexScores:     make([]float32, ctxSize/4+2),
		IndexAllowed:    make([]bool, ctxSize/4+2),
		FfnNormed:  make([]float32, NEmbd),
		RoutedXQ:   make([]byte, (NEmbd/QK_K)*BlockQ8KSize),
		RoutedMidQ: make([]byte, NExpertUsed*(NFFExp/QK_K)*BlockQ8KSize),
		RoutedOut:  make([]float32, NEmbd),
		SharedOut:  make([]float32, NEmbd),
		GateUp:     make([]float32, NFFExp),
		Post:       make([]float32, NHC),
		Comb:       make([]float32, NHC*NHC),
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
	copy(ds.AttnNormed, ds.CurHC[:NEmbd]) // extract stream 0
	rmsNorm(ds.AttnNormed, normW)

	// 2. Q projection: low-rank LoRA
	// attn_q_a: Q8_0 [NEmbd, NLoraQ] → qr[NLoraQ]
	matvecQ8_0(ds.QR, layer.AttnQA, ds.AttnNormed, NEmbd, NLoraQ)

	// RMSNorm on qr
	qrNormW := tensorF32Unsafe(layer.AttnQANorm)
	copy(ds.QRNorm, ds.QR)
	rmsNorm(ds.QRNorm, qrNormW)

	// attn_q_b: Q8_0 [NLoraQ, NHead*NHeadDim] → q[NHead*NHeadDim]
	matvecQ8_0(ds.Q, layer.AttnQB, ds.QRNorm, NLoraQ, NHead*NHeadDim)

	// Per-head RMSNorm
	for h := 0; h < NHead; h++ {
		head := ds.Q[h*NHeadDim : (h+1)*NHeadDim]
		rmsNormNoScale(head)
	}

	// Apply RoPE to Q tails
	freqBase := layerRoPEFreqBase(il)
	freqScale := layerRoPEFreqScale(il)
	ropeYaRNTailInplace(ds.Q, pos, NHead, NHeadDim, NRot, freqBase, freqScale, false)

	// 3. KV projection
	// attn_kv: Q8_0 [NEmbd, NHeadDim] → kv[NHeadDim]
	matvecQ8_0(ds.KV, layer.AttnKV, ds.AttnNormed, NEmbd, NHeadDim)

	// RMSNorm on KV
	kvNormW := tensorF32Unsafe(layer.AttnKVANorm)
	rmsNorm(ds.KV, kvNormW)

	// Apply RoPE to KV tail
	ropeYaRNTailInplace(ds.KV, pos, 1, NHeadDim, NRot, freqBase, freqScale, false)

	// FP8 quantize the non-RoPE portion for storage (reuse scratch to avoid alloc)
	copy(ds.KVCacheRow, ds.KV)
	fp8KVQuantizeInplace(ds.KVCacheRow, NHeadDim, NRot)

	// Push to cache
	cache.PushRawKV(ds.KVCacheRow)

	// 4. Compressor/indexer update
	ratio := cache.CompressRatio
	var compAllowed []bool
	if ratio > 0 && len(layer.CompressorKV) != 0 {
		if compressorDecodeOne(ds.CompOut, ds.CompKVCur, ds.CompScoreCur, ds.CompPooled,
			layer.CompressorKV, layer.CompressorGate, layer.CompressorAPE, layer.CompressorNorm,
			ds.AttnNormed, cache.CompStateKV, cache.CompStateScore,
			NHeadDim, ratio, il, pos, true) {
			cache.PushCompKV(ds.CompOut)
		}
		if ratio == 4 && len(layer.IndexerCompKV) != 0 {
			if compressorDecodeOne(ds.IndexCompOut, ds.IndexCompKVCur, ds.IndexCompScore, ds.IndexCompPooled,
				layer.IndexerCompKV, layer.IndexerCompGate, layer.IndexerCompAPE, layer.IndexerCompNorm,
				ds.AttnNormed, cache.IndexStateKV, cache.IndexStateScore,
				NIndexerHeadDim, ratio, il, pos, false) {
				cache.PushIndexCompKV(ds.IndexCompOut)
			}
			compAllowed = indexerAllowedDecodeOne(ds, layer, ds.AttnNormed, ds.QRNorm, cache.IndexCompKV, cache.NIndexComp, il, pos)
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

	sinks := tensorF32Unsafe(layer.AttnSinks)
	scale := float32(1.0 / math.Sqrt(float64(NHeadDim)))

	for h := 0; h < NHead; h++ {
		qHead := ds.Q[h*NHeadDim : (h+1)*NHeadDim]
		maxScore := sinks[h]

		for t := 0; t < nRaw; t++ {
			kvRow := cache.RawKV[t*NHeadDim : (t+1)*NHeadDim]
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
			kvRow := cache.CompKV[t*NHeadDim : (t+1)*NHeadDim]
			s := simd.Sdot(qHead, kvRow) * scale
			ds.AttnScore[nRaw+t] = s
			if s > maxScore {
				maxScore = s
			}
		}

		headOut := ds.Heads[h*NHeadDim : (h+1)*NHeadDim]
		for d := range headOut {
			headOut[d] = 0
		}
		denom := float32(math.Exp(float64(sinks[h] - maxScore)))

		for t := 0; t < nRaw; t++ {
			w := float32(math.Exp(float64(ds.AttnScore[t] - maxScore)))
			denom += w
			kvRow := cache.RawKV[t*NHeadDim : (t+1)*NHeadDim]
			simd.Saxpy(w, kvRow, headOut)
		}
		for t := 0; t < nComp; t++ {
			if ds.AttnScore[nRaw+t] <= -1e29 {
				continue
			}
			w := float32(math.Exp(float64(ds.AttnScore[nRaw+t] - maxScore)))
			denom += w
			kvRow := cache.CompKV[t*NHeadDim : (t+1)*NHeadDim]
			simd.Saxpy(w, kvRow, headOut)
		}

		if denom > 0 {
			inv := 1 / denom
			for d := range headOut {
				headOut[d] *= inv
			}
		}
		_ = nTotal
	}

	// 5. Output projection (de-RoPE → grouped LoRA)
	// Inverse RoPE on head tails
	ropeYaRNTailInplace(ds.Heads, pos, NHead, NHeadDim, NRot, freqBase, freqScale, true)

	// attn_output_a: Q8_0 [NHead*NValueDim, NLoraO] → tmp[NLoraO]
	matvecQ8_0Grouped(ds.TmpLoRA, layer.AttnOutputA, ds.Heads, NHead*NValueDim, NLoraO, NOutGroup)

	// attn_output_b: Q8_0 [NLoraO, NEmbd] → attn_out[NEmbd]
	matvecQ8_0(ds.AttnOut, layer.AttnOutputB, ds.TmpLoRA, NLoraO, NEmbd)
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
