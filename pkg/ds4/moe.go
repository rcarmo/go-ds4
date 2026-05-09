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

	// Dense layer (no experts): just run shared expert as standard FFN
	if len(layer.FfnGateInp) == 0 && len(layer.FfnGateExps) == 0 {
		sharedExpertForward(ds, layer)
		copy(ds.RoutedOut, ds.SharedOut)
		for i := range ds.SharedOut {
			ds.SharedOut[i] = 0
		}
		return
	}

	// 2. Quantize normed input to Q8_K for expert dot products
	QuantizeRowQ8K(ds.FfnNormed, ds.RoutedXQ)

	// 3. Expert routing — select top-K experts
	experts := routeExperts(ds, ds.FfnNormed, layer, il, tokenID, nExperts)
	if il < 3 {
	}

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

	strictGPU := false
	if ds.Engine != nil {
		if e, ok := ds.Engine.(*Engine); ok {
			strictGPU = e.StrictGPU
		}
	}

	// Start shared expert concurrently (independent of routed experts). In strict
	// GPU mode keep GPU dispatch serialized so kernel failures are attributable
	// and shared global CUDA buffers are not reused concurrently.
	var sharedWg sync.WaitGroup
	if !strictGPU {
		sharedWg.Add(1)
		go func() {
			defer sharedWg.Done()
			sharedExpertForward(ds, layer)
		}()
	}

	// GPU expert dispatch: cached experts on GPU, rest on CPU
	var gpuHandled []bool
	if ds.Engine != nil {
		if eng, ok := ds.Engine.(*Engine); ok {
			gpuHandled = eng.gpuExpertForward(ds, layer, experts, il)
		}
	}

	// CPU fallback for experts not handled by GPU
	needCPU := gpuHandled == nil
	if !needCPU {
		for _, h := range gpuHandled {
			if !h {
				needCPU = true
				break
			}
		}
	}

	if needCPU {
		if ds.Engine != nil {
			if e, ok := ds.Engine.(*Engine); ok && e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: routed expert CPU fallback refused at layer %d", il))
			}
		}
		// CPU: only run experts not already handled by GPU
		if len(experts) > 1 {
			// Parallel with preallocated per-expert scratch (no per-token allocs)
			cfg := ds.Cfg()
			ed := detectExpertDims(layer, cfg)
			ffnAligned := ((ed.outDim + QK_K - 1) / QK_K) * QK_K
			midQStride := (ffnAligned / QK_K) * BlockQ8KSize
			var wg sync.WaitGroup
			for i, exp := range experts {
				wg.Add(1)
				go func(idx int, e expertScore) {
					defer wg.Done()
					out := ds.ExpertOut[idx*cfg.NEmbd : (idx+1)*cfg.NEmbd]
					for j := range out {
						out[j] = 0
					}
					midQ := ds.ExpertMidQ[idx*midQStride : (idx+1)*midQStride]
					gate := ds.ExpertGate[idx*cfg.NFFExp : (idx+1)*cfg.NFFExp]
					up := ds.ExpertUp[idx*cfg.NFFExp : (idx+1)*cfg.NFFExp]
					expertForwardFast(out, ds.RoutedXQ, midQ, ds.Cfg(), gate, up, layer, e.idx, e.score, streamer, model, il)
				}(i, exp)
			}
			wg.Wait()
			for i := range experts {
				out := ds.ExpertOut[i*cfg.NEmbd : (i+1)*cfg.NEmbd]
				for j := 0; j < cfg.NEmbd; j++ {
					ds.RoutedOut[j] += out[j]
				}
			}
		} else if len(experts) == 1 {
			expertForward(ds, layer, experts[0].idx, experts[0].score, streamer, model, il)
		}
	} // end if !gpuDone

	// 5. Wait for/run shared expert
	if strictGPU {
		sharedExpertForward(ds, layer)
	} else {
		sharedWg.Wait()
	}

	// 6. Evict cold expert pages to stay within budget
	if budget != nil {
		budget.EvictColdExperts(il, activeIDs)
	}
}

// routeExperts selects the top-K experts for a token.
func routeExperts(ds *DecodeState, normed []float32, layer *LayerWeights, il, tokenID, nExperts int) []expertScore {
	cfg := ds.Cfg()
	// Router probabilities are sqrt(softplus(logit)); selected expert weights
	// are normalized only after top-k/hash selection (matches ds4.c).
	nExp := cfg.NExpert
	logits := ds.RouteLogits[:nExp]

	// V2 Lite uses F32 gate weights, V4 uses F16
	if len(layer.FfnGateInp) == nExp*cfg.NEmbd*4 {
		// F32 gate
		gateF32 := tensorF32Unsafe(layer.FfnGateInp)
		for e := 0; e < nExp; e++ {
			row := gateF32[e*cfg.NEmbd : (e+1)*cfg.NEmbd]
			dot := float32(0)
			for i := 0; i < cfg.NEmbd; i++ {
				dot += row[i] * normed[i]
			}
			logits[e] = dot
		}
	} else {
		// F16 gate (V4)
		gateU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&layer.FfnGateInp[0])), nExp*cfg.NEmbd)
		for e := 0; e < nExp; e++ {
			row := gateU16[e*cfg.NEmbd : (e+1)*cfg.NEmbd]
			logits[e] = DotF16(row, normed)
		}
	}

	// Reuse logits as unbiased router probabilities after projection.
	probs := logits
	for e := 0; e < nExp; e++ {
		probs[e] = float32(math.Sqrt(float64(softplusStable(logits[e]))))
	}

	// Hash routing (early V4 layers): table is [token, NExpertUsed] in memory.
	if layer.FfnGateTid2Eid != nil {
		nHash := nExperts
		if nHash > cfg.NExpertUsed {
			nHash = cfg.NExpertUsed
		}
		table := unsafe.Slice((*int32)(unsafe.Pointer(&layer.FfnGateTid2Eid[0])), cfg.NExpertUsed*cfg.NVocab)
		top := ds.RouteScores[:nHash]
		sum := float32(0)
		for k := 0; k < nHash; k++ {
			eid := int(table[tokenID*cfg.NExpertUsed+k])
			w := probs[eid]
			top[k] = expertScore{idx: eid, score: w}
			sum += w
		}
		if sum < 6.103515625e-5 {
			sum = 6.103515625e-5
		}
		for k := 0; k < nHash; k++ {
			top[k].score = top[k].score / sum * cfg.ExpertWeightScale
		}
		return top
	}

	// Later routing: selection is based on probs + optional bias, but final
	// expert weights use unbiased probs for the selected experts.
	selection := make([]float32, nExp)
	for e := 0; e < nExp; e++ {
		selection[e] = probs[e]
	}
	if layer.FfnExpProbsB != nil {
		bias := tensorF32Unsafe(layer.FfnExpProbsB)
		for e := 0; e < nExp; e++ {
			selection[e] += bias[e]
		}
	}

	scores := ds.RouteScores[:nExp]
	for i := range scores[:nExp] {
		scores[i] = expertScore{idx: i, score: selection[i]}
	}
	sort.Slice(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	topK := scores[:nExperts]
	sum := float32(0)
	for i := range topK {
		eid := topK[i].idx
		w := probs[eid]
		topK[i].score = w
		sum += w
	}
	if sum < 6.103515625e-5 {
		sum = 6.103515625e-5
	}
	for i := range topK {
		topK[i].score = topK[i].score / sum * cfg.ExpertWeightScale
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

// quantizeQ8KPadded pads input to QK_K alignment before quantizing.
func quantizeQ8KPadded(x []float32, out []byte) {
	n := len(x)
	aligned := ((n + QK_K - 1) / QK_K) * QK_K
	if aligned == n {
		QuantizeRowQ8K(x, out)
		return
	}
	padded := make([]float32, aligned)
	copy(padded, x)
	QuantizeRowQ8K(padded, out)
}

// expertForwardFast runs a single expert with pre-allocated buffers.
func expertForwardFast(out []float32, xQ8K, midQ []byte, cfg *ModelConfig,
	gate, up []float32,
	layer *LayerWeights, expertIdx int, weight float32,
	streamer *DiskStreamer, model *GGUFModel, il int) {

	ed := detectExpertDims(layer, cfg)
	gateRowBytes := ed.gateRowBytes
	upRowBytes := ed.upRowBytes
	downRowBytes := ed.downRowBytes
	ffnDim := ed.outDim
	if ffnDim == 0 {
		ffnDim = cfg.NFFExp
	}

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
		gateBase = layer.FfnGateExps[expertIdx*gateRowBytes*ffnDim:]
		upBase = layer.FfnUpExps[expertIdx*upRowBytes*ffnDim:]
		downBase = layer.FfnDownExps[expertIdx*downRowBytes*cfg.NEmbd:]
	}

	// IQ2_XXS gate+up projections
	for o := 0; o < ffnDim; o++ {
		if ed.gateIsIQ2 {
			gate[o] = VecDotIQ2XXSQ8K(cfg.NEmbd, gateBase[o*gateRowBytes:(o+1)*gateRowBytes], xQ8K)
		} else {
			gate[o] = VecDotQ2KQ8K(cfg.NEmbd, gateBase[o*gateRowBytes:(o+1)*gateRowBytes], xQ8K)
		}
		if ed.gateIsIQ2 {
			up[o] = VecDotIQ2XXSQ8K(cfg.NEmbd, upBase[o*upRowBytes:(o+1)*upRowBytes], xQ8K)
		} else {
			up[o] = VecDotQ2KQ8K(cfg.NEmbd, upBase[o*upRowBytes:(o+1)*upRowBytes], xQ8K)
		}
	}

	// DeepSeek V4 clamps routed expert gate/up before SwiGLU, then applies
	// router weight before Q8_K quantization (matches ds4.c).
	limit := cfg.SwiGLUClampExp
	if limit > 1.0e-6 {
		for i := 0; i < ffnDim; i++ {
			if gate[i] > limit {
				gate[i] = limit
			}
			if up[i] > limit {
				up[i] = limit
			} else if up[i] < -limit {
				up[i] = -limit
			}
		}
	}
	swiGLU(gate, gate, up)
	for i := 0; i < ffnDim; i++ {
		gate[i] *= weight
	}

	// Quantize hidden
	quantizeQ8KPadded(gate[:ffnDim], midQ)

	// Q2_K down projection
	for o := 0; o < cfg.NEmbd; o++ {
		if ed.downIsIQ4NL {
			out[o] += VecDotIQ4NLQ8K(ffnDim, downBase[o*downRowBytes:(o+1)*downRowBytes], midQ)
		} else if ed.downIsQ5K {
			out[o] += VecDotQ5KQ8K(ffnDim, downBase[o*downRowBytes:(o+1)*downRowBytes], midQ)
		} else {
			out[o] += VecDotQ2KQ8K(ffnDim, downBase[o*downRowBytes:(o+1)*downRowBytes], midQ)
		}
	}
}

// expertForwardInto runs a single routed expert into a provided output buffer.

// expertForward runs a single routed expert: gate·up (IQ2_XXS) → SwiGLU → down (Q2_K).
// If streamer is non-nil, expert weights are read from disk instead of mmap.
func expertForward(ds *DecodeState, layer *LayerWeights, expertIdx int, weight float32,
	streamer *DiskStreamer, model *GGUFModel, il int) {
	cfg := ds.Cfg()
	ed := detectExpertDims(layer, cfg)
	gateRowBytes := ed.gateRowBytes
	upRowBytes := ed.upRowBytes
	downRowBytes := ed.downRowBytes
	ffnDim := ed.outDim
	if ffnDim == 0 {
		ffnDim = cfg.NFFExp
	}

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
		gateBase = layer.FfnGateExps[expertIdx*gateRowBytes*ffnDim:]
		upBase = layer.FfnUpExps[expertIdx*upRowBytes*ffnDim:]
		downBase = layer.FfnDownExps[expertIdx*downRowBytes*cfg.NEmbd:]
	}

	gate := make([]float32, cfg.NFFExp)
	up := make([]float32, cfg.NFFExp)

	// Gate/Up: IQ2_XXS weight × Q8_K activation
	for o := 0; o < ffnDim; o++ {
		gateRow := gateBase[o*gateRowBytes : (o+1)*gateRowBytes]
		upRow := upBase[o*upRowBytes : (o+1)*upRowBytes]
		gate[o] = VecDotIQ2XXSQ8K(cfg.NEmbd, gateRow, ds.RoutedXQ)
		up[o] = VecDotIQ2XXSQ8K(cfg.NEmbd, upRow, ds.RoutedXQ)
	}

	limit := cfg.SwiGLUClampExp
	if limit > 1.0e-6 {
		for i := 0; i < ffnDim; i++ {
			if gate[i] > limit {
				gate[i] = limit
			}
			if up[i] > limit {
				up[i] = limit
			} else if up[i] < -limit {
				up[i] = -limit
			}
		}
	}

	// SwiGLU and apply router weight before Q8_K quantization.
	swiGLU(gate, gate, up)
	for i := 0; i < ffnDim; i++ {
		gate[i] *= weight
	}

	// Quantize hidden to Q8_K for down projection
	midQ := ds.RoutedMidQ // reuse scratch
	quantizeQ8KPadded(gate[:ffnDim], midQ)

	// Down projection: Q2_K [NEmbd, NFFExp] × Q8_K hidden
	for o := 0; o < cfg.NEmbd; o++ {
		downRow := downBase[o*downRowBytes : (o+1)*downRowBytes]
		ds.RoutedOut[o] += VecDotQ2KQ8K(ffnDim, downRow, midQ)
	}
}

// sharedExpertForward runs the always-active shared expert (Q8_0).
func sharedExpertForward(ds *DecodeState, layer *LayerWeights) {
	cfg := ds.Cfg()
	sharedFFN := cfg.NFFExp
	if len(layer.FfnGateShexp) > 0 {
		if d := detectOutDim(layer.FfnGateShexp, cfg.NEmbd); d > 0 {
			sharedFFN = d
		}
	}
	gate := make([]float32, sharedFFN)
	up := make([]float32, sharedFFN)
	if cfg.NHC > 0 {
		matvecQ8_0GPULayer(gate, layer.FfnGateShexp, ds.FfnNormed, cfg.NEmbd, sharedFFN, ds, "ffn_gate_shexp.weight")
		matvecQ8_0GPULayer(up, layer.FfnUpShexp, ds.FfnNormed, cfg.NEmbd, sharedFFN, ds, "ffn_up_shexp.weight")
		swiGLU(gate, gate, up)
		matvecQ8_0GPULayer(ds.SharedOut, layer.FfnDownShexp, gate, sharedFFN, cfg.NEmbd, ds, "ffn_down_shexp.weight")
		return
	}
	if ds.Engine != nil {
		if e, ok := ds.Engine.(*Engine); ok && e.StrictGPU {
			panic("strict GPU: V2 shared expert matvecAuto path has no GPU kernel")
		}
	}
	matvecAuto(gate, layer.FfnGateShexp, ds.FfnNormed, cfg.NEmbd, sharedFFN)
	matvecAuto(up, layer.FfnUpShexp, ds.FfnNormed, cfg.NEmbd, sharedFFN)
	swiGLU(gate, gate, up)
	matvecAuto(ds.SharedOut, layer.FfnDownShexp, gate, sharedFFN, cfg.NEmbd)
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
	model.PrefetchLayer(il)

	cfg := ds.Cfg()
	useHC := cfg.NHC > 0

	if useHC {
		attnResidual := ds.AttnResidual
		copy(attnResidual, ds.CurHC[:hcDim])
		hcPreFromState(
			ds.AttnCur, ds.Post, ds.Comb,
			attnResidual,
			layer.HCAttnFn, layer.HCAttnScale, layer.HCAttnBase,
			ds.HCFlat, ds.HCMix,
		)
		layerAttnDecode(ds, layer, cache, model, pos, il)
		hcPostOne(ds.NextHC, ds.AttnOut, attnResidual, ds.Post, ds.Comb)
		ffnResidual := ds.FfnResidual
		copy(ffnResidual, ds.NextHC[:hcDim])
		hcPreFromState(
			ds.FfnCur, ds.Post, ds.Comb,
			ffnResidual,
			layer.HCFfnFn, layer.HCFfnScale, layer.HCFfnBase,
			ds.HCFlat, ds.HCMix,
		)
		layerFFNDecode(ds, layer, model, budget, streamer, il, tokenID, nExperts)
		hcPostSumOne(ds.CurHC, ds.RoutedOut, ds.SharedOut, ffnResidual, ds.Post, ds.Comb, ds.HCSumTmp)
	} else {
		// Standard residual path (V2 Lite): attention residual, then FFN residual.
		copy(ds.AttnCur[:cfg.NEmbd], ds.CurHC[:cfg.NEmbd])
		layerAttnDecode(ds, layer, cache, model, pos, il)
		for i := 0; i < cfg.NEmbd; i++ {
			ds.CurHC[i] += ds.AttnOut[i]
		}
		copy(ds.FfnCur[:cfg.NEmbd], ds.CurHC[:cfg.NEmbd])
		layerFFNDecode(ds, layer, model, budget, streamer, il, tokenID, nExperts)
		for i := 0; i < cfg.NEmbd; i++ {
			ds.CurHC[i] += ds.RoutedOut[i] + ds.SharedOut[i]
		}
	}
}

// outputLogits computes the LM head: HC collapse → RMSNorm → Q8_0 matmul → logits.
func outputLogits(ds *DecodeState, logits []float32, hcState []float32, w *Weights) {
	cfg := ds.Cfg()

	var collapsed []float32
	if cfg.NHC > 0 && len(w.OutputHCBase) > 0 {
		hcBase := tensorF32Unsafe(w.OutputHCBase)
		hcScale := tensorF32Unsafe(w.OutputHCScale)
		hcFnU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&w.OutputHCFn[0])), hcDim*NHC)
		flat := ds.OutFlat
		copy(flat, hcState[:hcDim])
		rmsNormNoScale(flat)
		hcWeights := ds.OutHCWeights
		for j := 0; j < NHC; j++ {
			row := hcFnU16[j*hcDim : (j+1)*hcDim]
			hcWeights[j] = sigmoid(hcScale[0]*DotF16(row, flat)+hcBase[j]) + HCEps
		}
		collapsed = ds.OutCollapsed
		for i := range collapsed {
			collapsed[i] = 0
		}
		for s := 0; s < NHC; s++ {
			wt := hcWeights[s]
			for i := 0; i < NEmbd; i++ {
				collapsed[i] += wt * hcState[s*NEmbd+i]
			}
		}
	} else {
		collapsed = make([]float32, cfg.NEmbd)
		copy(collapsed, hcState[:cfg.NEmbd])
	}

	outputNormW := tensorF32Unsafe(w.OutputNorm)
	rmsNorm(collapsed, outputNormW[:cfg.NEmbd])
	if cfg.NHC > 0 {
		matvecQ8_0GPU(logits, w.Output, collapsed, cfg.NEmbd, cfg.NVocab, ds.Engine, "output.weight")
		return
	}
	if ds.Engine != nil {
		if e, ok := ds.Engine.(*Engine); ok && e.StrictGPU {
			panic("strict GPU: V2 output matvecAuto path has no GPU kernel")
		}
	}
	matvecAuto(logits, w.Output, collapsed, cfg.NEmbd, cfg.NVocab)
}

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
