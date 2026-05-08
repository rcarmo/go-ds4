package ds4

import (
	"math"
	"unsafe"
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

	// FP8 quantize the non-RoPE portion for storage
	kvForCache := make([]float32, NHeadDim)
	copy(kvForCache, ds.KV)
	fp8KVQuantizeInplace(kvForCache, NHeadDim, NRot)

	// Push to cache
	cache.PushRawKV(kvForCache)

	// 4. Attention scoring over SWA window
	nRaw := cache.NRaw
	if nRaw > cache.CapRaw {
		nRaw = cache.CapRaw
	}

	scale := float32(1.0 / math.Sqrt(float64(NHeadDim)))

	// For each Q head, score against all raw KV rows
	for h := 0; h < NHead; h++ {
		qHead := ds.Q[h*NHeadDim : (h+1)*NHeadDim]
		bestSum := float32(0)
		for t := 0; t < nRaw; t++ {
			idx := t % cache.CapRaw
			kvRow := cache.RawKV[idx*NHeadDim : (idx+1)*NHeadDim]
			// Dot product Q·KV
			dot := float32(0)
			for d := 0; d < NHeadDim; d++ {
				dot += qHead[d] * kvRow[d]
			}
			ds.AttnScore[t] = dot * scale
			bestSum += dot * scale
		}
		_ = bestSum // will be used after softmax

		// Softmax over scores
		softmax(ds.AttnScore[:nRaw])

		// Weighted sum of KV rows → head output
		headOut := ds.Heads[h*NHeadDim : (h+1)*NHeadDim]
		for d := range headOut {
			headOut[d] = 0
		}
		for t := 0; t < nRaw; t++ {
			idx := t % cache.CapRaw
			kvRow := cache.RawKV[idx*NHeadDim : (idx+1)*NHeadDim]
			w := ds.AttnScore[t]
			for d := 0; d < NHeadDim; d++ {
				headOut[d] += w * kvRow[d]
			}
		}
	}

	// 5. Output projection (de-RoPE → grouped LoRA)
	// Inverse RoPE on head tails
	ropeYaRNTailInplace(ds.Heads, pos, NHead, NHeadDim, NRot, freqBase, freqScale, true)

	// attn_output_a: Q8_0 [NHead*NValueDim, NLoraO] → tmp[NLoraO]
	tmpLoRA := make([]float32, NLoraO)
	matvecQ8_0Grouped(tmpLoRA, layer.AttnOutputA, ds.Heads, NHead*NValueDim, NLoraO, NOutGroup)

	// attn_output_b: Q8_0 [NLoraO, NEmbd] → attn_out[NEmbd]
	matvecQ8_0(ds.AttnOut, layer.AttnOutputB, tmpLoRA, NLoraO, NEmbd)
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
