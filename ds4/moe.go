package ds4

import (
	"math"
	"sort"
	"unsafe"
)

// expertScore holds an expert index and its routing score.
type expertScore struct {
	idx   int
	score float32
}

// layerFFNDecode runs the MoE FFN for a single decode token.
func layerFFNDecode(
	ds *DecodeState,
	layer *LayerWeights,
	model *GGUFModel,
	il, tokenID int,
) {
	// 1. RMSNorm FFN input
	normW := tensorF32Unsafe(layer.FfnNorm)
	// Extract stream 0 from current HC
	copy(ds.FfnNormed, ds.CurHC[:NEmbd])
	rmsNorm(ds.FfnNormed, normW)

	// 2. Quantize normed input to Q8_K for expert dot products
	QuantizeRowQ8K(ds.FfnNormed, ds.RoutedXQ)

	// 3. Expert routing — select top-K experts
	experts := routeExperts(ds.FfnNormed, layer, il, tokenID)

	// 4. Run routed experts
	for i := range ds.RoutedOut {
		ds.RoutedOut[i] = 0
	}

	for _, exp := range experts {
		expertForward(ds, layer, exp.idx, exp.score)
	}

	// 5. Run shared expert
	sharedExpertForward(ds, layer)

	// 6. HC post: combine routed + shared outputs
}

// routeExperts selects the top-K experts for a token.
func routeExperts(normed []float32, layer *LayerWeights, il, tokenID int) []expertScore {
	// Check for hash routing (3 hash layers)
	if layer.FfnGateTid2Eid != nil {
		return hashRouteExperts(layer, tokenID)
	}

	// Standard routing: F16 matmul normed → 256 expert logits
	logits := make([]float32, NExpert)
	gateU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&layer.FfnGateInp[0])), NExpert*NEmbd)
	for e := 0; e < NExpert; e++ {
		row := gateU16[e*NEmbd : (e+1)*NEmbd]
		logits[e] = DotF16(row, normed)
	}

	// Add bias
	if layer.FfnExpProbsB != nil {
		bias := tensorF32Unsafe(layer.FfnExpProbsB)
		for e := 0; e < NExpert; e++ {
			logits[e] += bias[e]
		}
	}

	// Softmax
	softmax(logits)

	// Scale
	for e := range logits {
		logits[e] *= ExpertWeightScale
	}

	// Top-K selection
	scores := make([]expertScore, NExpert)
	for i := range scores {
		scores[i] = expertScore{idx: i, score: logits[i]}
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Normalize top-K weights
	topK := scores[:NExpertUsed]
	sum := float32(0)
	for _, s := range topK {
		sum += s.score
	}
	if sum > 0 {
		inv := 1.0 / sum
		for i := range topK {
			topK[i].score *= inv
		}
	}

	return topK
}

// hashRouteExperts uses the token-ID → expert-ID lookup table.
func hashRouteExperts(layer *LayerWeights, tokenID int) []expertScore {
	// tid2eid: [NExpertUsed, NVocab] int32
	table := unsafe.Slice((*int32)(unsafe.Pointer(&layer.FfnGateTid2Eid[0])),
		NExpertUsed*NVocab)

	experts := make([]expertScore, NExpertUsed)
	for k := 0; k < NExpertUsed; k++ {
		eid := int(table[k*NVocab+tokenID])
		experts[k] = expertScore{
			idx:   eid,
			score: 1.0 / float32(NExpertUsed), // equal weights for hash routing
		}
	}
	return experts
}

// expertForward runs a single routed expert: gate·up (IQ2_XXS) → SwiGLU → down (Q2_K).
func expertForward(ds *DecodeState, layer *LayerWeights, expertIdx int, weight float32) {
	// Gate and up projections are IQ2_XXS: [NFFExp, NExpert*NEmbd]
	// Each expert's slice: [NFFExp, NEmbd] starting at expertIdx * rowBytes
	gateRowBytes := (NEmbd / QK_K) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / QK_K) * BlockQ2KSize

	gateBase := layer.FfnGateExps[expertIdx*gateRowBytes*NFFExp:]
	upBase := layer.FfnUpExps[expertIdx*upRowBytes*NFFExp:]

	gate := make([]float32, NFFExp)
	up := make([]float32, NFFExp)

	// Gate/Up: IQ2_XXS weight × Q8_K activation
	for o := 0; o < NFFExp; o++ {
		gateRow := gateBase[o*gateRowBytes : (o+1)*gateRowBytes]
		upRow := upBase[o*upRowBytes : (o+1)*upRowBytes]
		gate[o] = VecDotIQ2XXSQ8K(NEmbd, gateRow, ds.RoutedXQ)
		up[o] = VecDotIQ2XXSQ8K(NEmbd, upRow, ds.RoutedXQ)
	}

	// SwiGLU: dst = silu(gate) * up
	swiGLU(gate, gate, up)

	// Clamp
	for i := range gate {
		if gate[i] > SwiGLUClampExp {
			gate[i] = SwiGLUClampExp
		} else if gate[i] < -SwiGLUClampExp {
			gate[i] = -SwiGLUClampExp
		}
	}

	// Quantize hidden to Q8_K for down projection
	midQ := ds.RoutedMidQ // reuse scratch
	QuantizeRowQ8K(gate, midQ)

	// Down projection: Q2_K [NEmbd, NFFExp] × Q8_K hidden
	downBase := layer.FfnDownExps[expertIdx*downRowBytes*NEmbd:]
	for o := 0; o < NEmbd; o++ {
		downRow := downBase[o*downRowBytes : (o+1)*downRowBytes]
		ds.RoutedOut[o] += weight * VecDotQ2KQ8K(NFFExp, downRow, midQ)
	}
}

// sharedExpertForward runs the always-active shared expert (Q8_0).
func sharedExpertForward(ds *DecodeState, layer *LayerWeights) {
	// Gate: Q8_0 [NFFExp, NEmbd]
	gate := make([]float32, NFFExp)
	up := make([]float32, NFFExp)

	matvecQ8_0(gate, layer.FfnGateShexp, ds.FfnNormed, NEmbd, NFFExp)
	matvecQ8_0(up, layer.FfnUpShexp, ds.FfnNormed, NEmbd, NFFExp)

	// SwiGLU
	swiGLU(gate, gate, up)

	// Down: Q8_0 [NEmbd, NFFExp]
	matvecQ8_0(ds.SharedOut, layer.FfnDownShexp, gate, NFFExp, NEmbd)
}

// layerForwardDecode runs one full transformer layer for a single decode token.
func layerForwardDecode(
	ds *DecodeState,
	layer *LayerWeights,
	cache *LayerCache,
	model *GGUFModel,
	pos, il, tokenID int,
) {
	// Save residual HC for attention sublayer
	attnResidual := make([]float32, hcDim)
	copy(attnResidual, ds.CurHC[:hcDim])

	// HC pre → attention input
	hcPreFromState(
		ds.AttnNormed, ds.Post, ds.Comb,
		attnResidual,
		layer.HCAttnFn, layer.HCAttnScale, layer.HCAttnBase,
	)

	// Run MLA attention
	layerAttnDecode(ds, layer, cache, model, pos, il)

	// HC post (attention output → HC state)
	hcPostOne(ds.NextHC, ds.AttnOut, attnResidual, ds.Post, ds.Comb)

	// Save FFN residual
	ffnResidual := make([]float32, hcDim)
	copy(ffnResidual, ds.NextHC[:hcDim])

	// HC pre → FFN input
	hcPreFromState(
		ds.FfnNormed, ds.Post, ds.Comb,
		ffnResidual,
		layer.HCFfnFn, layer.HCFfnScale, layer.HCFfnBase,
	)

	// Run MoE FFN
	layerFFNDecode(ds, layer, model, il, tokenID)

	// HC post (routed + shared → HC state)
	hcPostSumOne(ds.CurHC, ds.RoutedOut, ds.SharedOut, ffnResidual, ds.Post, ds.Comb)
}

// outputLogits computes the LM head: HC collapse → RMSNorm → Q8_0 matmul → logits.
func outputLogits(logits []float32, hcState []float32, w *Weights) {
	// 1. HC head collapse: sigmoid-weighted sum of 4 streams
	hcBase := tensorF32Unsafe(w.OutputHCBase)
	hcScale := tensorF32Unsafe(w.OutputHCScale)
	hcFnU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&w.OutputHCFn[0])), hcDim*NHC)

	// Project HC flat → NHC weights
	flat := make([]float32, hcDim)
	copy(flat, hcState[:hcDim])
	rmsNormNoScale(flat)

	hcWeights := make([]float32, NHC)
	for j := 0; j < NHC; j++ {
		row := hcFnU16[j*hcDim : (j+1)*hcDim]
		hcWeights[j] = sigmoid(hcScale[0]*DotF16(row, flat) + hcBase[j])
	}

	// Weighted sum: collapsed[NEmbd] = Σ hcWeights[s] * hcState[s*NEmbd:]
	collapsed := make([]float32, NEmbd)
	for s := 0; s < NHC; s++ {
		wt := hcWeights[s]
		for i := 0; i < NEmbd; i++ {
			collapsed[i] += wt * hcState[s*NEmbd+i]
		}
	}

	// 2. Output RMSNorm
	outputNormW := tensorF32Unsafe(w.OutputNorm)
	rmsNorm(collapsed, outputNormW)

	// 3. Output projection: Q8_0 [NEmbd, NVocab]
	matvecQ8_0(logits, w.Output, collapsed, NEmbd, NVocab)
}

// Argmax returns the index of the maximum value.
func Argmax(x []float32) int {
	best := 0
	bestVal := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > bestVal {
			bestVal = x[i]
			best = i
		}
	}
	return best
}

// SampleTopK samples from top-k logits with temperature.
func SampleTopK(logits []float32, temperature float32, topK int) int {
	if temperature <= 0 || topK <= 1 {
		return Argmax(logits)
	}

	type scored struct {
		idx int
		val float32
	}
	all := make([]scored, len(logits))
	for i, v := range logits {
		all[i] = scored{i, v}
	}
	sort.Slice(all, func(i, j int) bool { return all[i].val > all[j].val })

	if topK > len(all) {
		topK = len(all)
	}
	top := all[:topK]

	// Temperature + softmax
	maxV := top[0].val
	sum := float32(0)
	for i := range top {
		e := float32(math.Exp(float64((top[i].val - maxV) / temperature)))
		top[i].val = e
		sum += e
	}

	// Sample (using a simple linear scan — good enough for small topK)
	// In production, use a proper RNG
	r := float32(0.5) * sum // deterministic for now
	cumul := float32(0)
	for _, s := range top {
		cumul += s.val
		if cumul >= r {
			return s.idx
		}
	}
	return top[len(top)-1].idx
}
