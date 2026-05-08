package ds4

import (
	"fmt"
	"math"
	"sort"
	"sync"
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
	budget *MemoryBudget,
	streamer *DiskStreamer,
	il, tokenID int,
	nExperts int,
) {
	// 1. RMSNorm FFN input (from HC-pre output)
	normW := tensorF32Unsafe(layer.FfnNorm)
	copy(ds.FfnNormed, ds.FfnCur)
	rmsNorm(ds.FfnNormed, normW)

	// 2. Quantize normed input to Q8_K for expert dot products
	QuantizeRowQ8K(ds.FfnNormed, ds.RoutedXQ)

	// 3. Expert routing — select top-K experts
	experts := routeExperts(ds, ds.FfnNormed, layer, il, tokenID, nExperts)

	// Prefetch selected expert pages
	var activeIDsBuf [NExpertUsed]int
	activeIDs := activeIDsBuf[:len(experts)]
	for i, e := range experts {
		activeIDs[i] = e.idx
	}
	model.PrefetchExperts(il, activeIDs)

	// 4. Run routed experts — try GPU first, fall back to CPU parallel
	for i := range ds.RoutedOut {
		ds.RoutedOut[i] = 0
	}

	// Start shared expert concurrently (independent of routed experts)
	var sharedWg sync.WaitGroup
	sharedWg.Add(1)
	go func() {
		defer sharedWg.Done()
		sharedExpertForward(ds, layer)
	}()

	// GPU expert dispatch disabled: PCIe weight streaming overhead (~1.1 GB/token)
	// exceeds GPU kernel speedup. CPU mmap page cache is faster for sparse expert access.
	// Keeping the infrastructure for future use with larger VRAM or NVLink.
	gpuDone := false

	if !gpuDone {
		// CPU fallback: parallel experts
		if len(experts) > 1 {
			// Parallel with preallocated per-expert scratch (no per-token allocs)
			midQStride := (NFFExp / QK_K) * BlockQ8KSize
			var wg sync.WaitGroup
			for i, exp := range experts {
				wg.Add(1)
				go func(idx int, e expertScore) {
					defer wg.Done()
					out := ds.ExpertOut[idx*NEmbd : (idx+1)*NEmbd]
					for j := range out {
						out[j] = 0
					}
					midQ := ds.ExpertMidQ[idx*midQStride : (idx+1)*midQStride]
					gate := ds.ExpertGate[idx*NFFExp : (idx+1)*NFFExp]
					up := ds.ExpertUp[idx*NFFExp : (idx+1)*NFFExp]
					expertForwardFast(out, ds.RoutedXQ, midQ, gate, up, layer, e.idx, e.score, streamer, model, il)
				}(i, exp)
			}
			wg.Wait()
			for i := range experts {
				out := ds.ExpertOut[i*NEmbd : (i+1)*NEmbd]
				for j := 0; j < NEmbd; j++ {
					ds.RoutedOut[j] += out[j]
				}
			}
		} else if len(experts) == 1 {
			expertForward(ds, layer, experts[0].idx, experts[0].score, streamer, model, il)
		}
	} // end if !gpuDone

	// 5. Wait for shared expert
	sharedWg.Wait()

	// 6. Evict cold expert pages to stay within budget
	if budget != nil {
		budget.EvictColdExperts(il, activeIDs)
	}
}

// routeExperts selects the top-K experts for a token.
func routeExperts(ds *DecodeState, normed []float32, layer *LayerWeights, il, tokenID, nExperts int) []expertScore {
	// Check for hash routing (3 hash layers)
	if layer.FfnGateTid2Eid != nil {
		table := unsafe.Slice((*int32)(unsafe.Pointer(&layer.FfnGateTid2Eid[0])), NExpertUsed*NVocab)
		top := ds.RouteScores[:NExpertUsed]
		w := float32(1.0 / float32(NExpertUsed))
		for k := 0; k < NExpertUsed; k++ {
			top[k] = expertScore{idx: int(table[k*NVocab+tokenID]), score: w}
		}
		return top
	}

	// Standard routing: F16 matmul normed → 256 expert logits
	logits := ds.RouteLogits
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
	scores := ds.RouteScores
	for i := range scores {
		scores[i] = expertScore{idx: i, score: logits[i]}
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	// Normalize top-K weights
	topK := scores[:nExperts]
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

// expertForwardFast runs a single expert with pre-allocated buffers.
func expertForwardFast(out []float32, xQ8K, midQ []byte,
	gate, up []float32,
	layer *LayerWeights, expertIdx int, weight float32,
	streamer *DiskStreamer, model *GGUFModel, il int) {

	gateRowBytes := (NEmbd / QK_K) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / QK_K) * BlockQ2KSize

	var gateBase, upBase, downBase []byte

	if streamer != nil {
		prefix := fmt.Sprintf("blk.%d.", il)
		var err error
		gateBase, err = streamer.ReadExpertTensor(model.Tensors[prefix+"ffn_gate_exps.weight"], expertIdx)
		if err != nil {
			return
		}
		upBase, err = streamer.ReadExpertTensor(model.Tensors[prefix+"ffn_up_exps.weight"], expertIdx)
		if err != nil {
			return
		}
		downBase, err = streamer.ReadExpertTensor(model.Tensors[prefix+"ffn_down_exps.weight"], expertIdx)
		if err != nil {
			return
		}
		defer func() {
			streamer.ReturnBuffer(gateBase)
			streamer.ReturnBuffer(upBase)
			streamer.ReturnBuffer(downBase)
		}()
	} else {
		gateBase = layer.FfnGateExps[expertIdx*gateRowBytes*NFFExp:]
		upBase = layer.FfnUpExps[expertIdx*upRowBytes*NFFExp:]
		downBase = layer.FfnDownExps[expertIdx*downRowBytes*NEmbd:]
	}

	// IQ2_XXS gate+up projections
	for o := 0; o < NFFExp; o++ {
		gate[o] = VecDotIQ2XXSQ8K(NEmbd, gateBase[o*gateRowBytes:(o+1)*gateRowBytes], xQ8K)
		up[o] = VecDotIQ2XXSQ8K(NEmbd, upBase[o*upRowBytes:(o+1)*upRowBytes], xQ8K)
	}

	// SwiGLU
	swiGLU(gate, gate, up)

	// Quantize hidden
	QuantizeRowQ8K(gate, midQ)

	// Q2_K down projection
	for o := 0; o < NEmbd; o++ {
		out[o] += weight * VecDotQ2KQ8K(NFFExp, downBase[o*downRowBytes:(o+1)*downRowBytes], midQ)
	}
}

// expertForwardInto runs a single routed expert into a provided output buffer.
// Thread-safe: uses provided scratch buffers instead of shared DecodeState.
func expertForwardInto(out []float32, xQ8K, midQ []byte,
	layer *LayerWeights, expertIdx int, weight float32,
	streamer *DiskStreamer, model *GGUFModel, il int) {

	gateRowBytes := (NEmbd / QK_K) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / QK_K) * BlockQ2KSize

	var gateBase, upBase, downBase []byte

	if streamer != nil {
		prefix := fmt.Sprintf("blk.%d.", il)
		gateTensor := model.Tensors[prefix+"ffn_gate_exps.weight"]
		upTensor := model.Tensors[prefix+"ffn_up_exps.weight"]
		downTensor := model.Tensors[prefix+"ffn_down_exps.weight"]
		var err error
		gateBase, err = streamer.ReadExpertTensor(gateTensor, expertIdx)
		if err != nil {
			return
		}
		upBase, err = streamer.ReadExpertTensor(upTensor, expertIdx)
		if err != nil {
			return
		}
		downBase, err = streamer.ReadExpertTensor(downTensor, expertIdx)
		if err != nil {
			return
		}
		defer func() {
			streamer.ReturnBuffer(gateBase)
			streamer.ReturnBuffer(upBase)
			streamer.ReturnBuffer(downBase)
		}()
	} else {
		gateBase = layer.FfnGateExps[expertIdx*gateRowBytes*NFFExp:]
		upBase = layer.FfnUpExps[expertIdx*upRowBytes*NFFExp:]
		downBase = layer.FfnDownExps[expertIdx*downRowBytes*NEmbd:]
	}

	gate := make([]float32, NFFExp)
	up := make([]float32, NFFExp)

	for o := 0; o < NFFExp; o++ {
		gateRow := gateBase[o*gateRowBytes : (o+1)*gateRowBytes]
		upRow := upBase[o*upRowBytes : (o+1)*upRowBytes]
		gate[o] = VecDotIQ2XXSQ8K(NEmbd, gateRow, xQ8K)
		up[o] = VecDotIQ2XXSQ8K(NEmbd, upRow, xQ8K)
	}

	swiGLU(gate, gate, up)

	for i := range gate {
		if gate[i] > SwiGLUClampExp {
			gate[i] = SwiGLUClampExp
		}
		if gate[i] < -SwiGLUClampExp {
			gate[i] = -SwiGLUClampExp
		}
	}

	QuantizeRowQ8K(gate, midQ)

	for o := 0; o < NEmbd; o++ {
		downRow := downBase[o*downRowBytes : (o+1)*downRowBytes]
		out[o] += weight * VecDotQ2KQ8K(NFFExp, downRow, midQ)
	}
}

// expertForward runs a single routed expert: gate·up (IQ2_XXS) → SwiGLU → down (Q2_K).
// If streamer is non-nil, expert weights are read from disk instead of mmap.
func expertForward(ds *DecodeState, layer *LayerWeights, expertIdx int, weight float32,
	streamer *DiskStreamer, model *GGUFModel, il int) {
	gateRowBytes := (NEmbd / QK_K) * BlockIQ2XXSSize
	upRowBytes := gateRowBytes
	downRowBytes := (NFFExp / QK_K) * BlockQ2KSize

	var gateBase, upBase, downBase []byte

	if streamer != nil {
		// Stream from disk: read only this expert's slice
		prefix := fmt.Sprintf("blk.%d.", il)
		gateTensor := model.Tensors[prefix+"ffn_gate_exps.weight"]
		upTensor := model.Tensors[prefix+"ffn_up_exps.weight"]
		downTensor := model.Tensors[prefix+"ffn_down_exps.weight"]

		var err error
		gateBase, err = streamer.ReadExpertTensor(gateTensor, expertIdx)
		if err != nil {
			return // silently skip on read error
		}
		upBase, err = streamer.ReadExpertTensor(upTensor, expertIdx)
		if err != nil {
			return
		}
		downBase, err = streamer.ReadExpertTensor(downTensor, expertIdx)
		if err != nil {
			return
		}
		defer func() {
			streamer.ReturnBuffer(gateBase)
			streamer.ReturnBuffer(upBase)
			streamer.ReturnBuffer(downBase)
		}()
	} else {
		// mmap path: slice into the contiguous expert tensor
		gateBase = layer.FfnGateExps[expertIdx*gateRowBytes*NFFExp:]
		upBase = layer.FfnUpExps[expertIdx*upRowBytes*NFFExp:]
		downBase = layer.FfnDownExps[expertIdx*downRowBytes*NEmbd:]
	}

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
	for o := 0; o < NEmbd; o++ {
		downRow := downBase[o*downRowBytes : (o+1)*downRowBytes]
		ds.RoutedOut[o] += weight * VecDotQ2KQ8K(NFFExp, downRow, midQ)
	}
}

// sharedExpertForward runs the always-active shared expert (Q8_0).
func sharedExpertForward(ds *DecodeState, layer *LayerWeights) {
	gate := ds.SharedGate
	up := ds.SharedUp
	matvecQ8_0GPULayer(gate, layer.FfnGateShexp, ds.FfnNormed, NEmbd, NFFExp, ds, "ffn_gate_shexp.weight")
	matvecQ8_0GPULayer(up, layer.FfnUpShexp, ds.FfnNormed, NEmbd, NFFExp, ds, "ffn_up_shexp.weight")
	swiGLU(gate, gate, up)
	matvecQ8_0GPULayer(ds.SharedOut, layer.FfnDownShexp, gate, NFFExp, NEmbd, ds, "ffn_down_shexp.weight")
}

// layerForwardDecode runs one full transformer layer for a single decode token.
func layerForwardDecode(
	ds *DecodeState,
	layer *LayerWeights,
	cache *LayerCache,
	model *GGUFModel,
	budget *MemoryBudget,
	streamer *DiskStreamer,
	pos, il, tokenID, nExperts int,
) {
	ds.LayerIdx = il
	// Prefetch non-expert weights for this layer
	model.PrefetchLayer(il)

	// Save residual HC for attention sublayer
	attnResidual := ds.AttnResidual
	copy(attnResidual, ds.CurHC[:hcDim])

	// HC pre → attention input
	hcPreFromState(
		ds.AttnCur, ds.Post, ds.Comb,
		attnResidual,
		layer.HCAttnFn, layer.HCAttnScale, layer.HCAttnBase,
		ds.HCFlat, ds.HCMix,
	)

	// Run MLA attention
	layerAttnDecode(ds, layer, cache, model, pos, il)

	// HC post (attention output → HC state)
	hcPostOne(ds.NextHC, ds.AttnOut, attnResidual, ds.Post, ds.Comb)

	// Save FFN residual
	ffnResidual := ds.FfnResidual
	copy(ffnResidual, ds.NextHC[:hcDim])

	// HC pre → FFN input
	hcPreFromState(
		ds.FfnCur, ds.Post, ds.Comb,
		ffnResidual,
		layer.HCFfnFn, layer.HCFfnScale, layer.HCFfnBase,
		ds.HCFlat, ds.HCMix,
	)

	// Run MoE FFN
	layerFFNDecode(ds, layer, model, budget, streamer, il, tokenID, nExperts)

	// HC post (routed + shared → HC state)
	hcPostSumOne(ds.CurHC, ds.RoutedOut, ds.SharedOut, ffnResidual, ds.Post, ds.Comb, ds.HCSumTmp)
}

// outputLogits computes the LM head: HC collapse → RMSNorm → Q8_0 matmul → logits.
func outputLogits(ds *DecodeState, logits []float32, hcState []float32, w *Weights) {
	hcBase := tensorF32Unsafe(w.OutputHCBase)
	hcScale := tensorF32Unsafe(w.OutputHCScale)
	hcFnU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&w.OutputHCFn[0])), hcDim*NHC)

	flat := ds.OutFlat
	copy(flat, hcState[:hcDim])
	rmsNormNoScale(flat)

	hcWeights := ds.OutHCWeights
	for j := 0; j < NHC; j++ {
		row := hcFnU16[j*hcDim : (j+1)*hcDim]
		hcWeights[j] = sigmoid(hcScale[0]*DotF16(row, flat) + hcBase[j])
	}

	collapsed := ds.OutCollapsed
	for i := range collapsed {
		collapsed[i] = 0
	}
	for s := 0; s < NHC; s++ {
		wt := hcWeights[s]
		for i := 0; i < NEmbd; i++ {
			collapsed[i] += wt * hcState[s*NEmbd+i]
		}
	}

	outputNormW := tensorF32Unsafe(w.OutputNorm)
	rmsNorm(collapsed, outputNormW)
	// 3. Output projection: Q8_0 [NEmbd, NVocab] — GPU-accelerated if available
	matvecQ8_0GPU(logits, w.Output, collapsed, NEmbd, NVocab, ds.Engine, "output.weight")
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
