//go:build darwin && cgo && metal

package ds4

/*
#cgo CFLAGS: -I${SRCDIR}/../.. -O3 -ffast-math -mcpu=native -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal
#include <stdint.h>
#include <stdlib.h>
#include "ds4_metal.h"
*/
import "C"

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"
	"unsafe"
)

type metalTensor struct {
	ptr   *C.ds4_metal_tensor
	bytes uint64
}

// MetalEngine is the optional macOS backend that reuses the C/Objective-C
// Metal host and shaders while keeping Go in charge of token/session flow.
type MetalEngine struct {
	mu        sync.Mutex
	ready     bool
	strict    bool
	enableMoE bool
	streaming bool
	q8MinRows int
	model     *GGUFModel
	mapPtr    unsafe.Pointer
	modelSize uint64

	x, out          metalTensor
	moeX, moeGate   metalTensor
	moeUp, moeMid   metalTensor
	moeOut, moeExps metalTensor
	moeSelected     metalTensor
	moeWeights      metalTensor

	prefill metalPrefillGraph
}

type metalPrefillGraph struct {
	allocated bool
	ctxSize   int
	capTokens int
	rawCap    int
	compCap   int

	tokens            metalTensor
	curHC, nextHC     metalTensor
	flatHC, hcMix     metalTensor
	hcSplit           metalTensor
	attnCur, attnNorm metalTensor
	qr, qrNorm, q     metalTensor
	kvRaw, kv         metalTensor
	compKV, compSC    metalTensor
	indexerQ          metalTensor
	indexerWeights    metalTensor
	indexerScores     metalTensor
	topK              metalTensor
	heads             metalTensor
	attnLow, attnOut  metalTensor
	groupTmp, lowTmp  metalTensor
	afterAttnHC       metalTensor
	ffnCur, ffnNorm   metalTensor
	sharedGate        metalTensor
	sharedUp          metalTensor
	sharedMid         metalTensor
	sharedOut         metalTensor
	routerLogits      metalTensor
	routerProbs       metalTensor
	routerSelected    metalTensor
	routerWeights     metalTensor
	routedGate        metalTensor
	routedUp          metalTensor
	routedMid         metalTensor
	routedDown        metalTensor
	routedOut         metalTensor
	outputPre         metalTensor
	outputWeights     metalTensor
	outputEmbd        metalTensor
	outputNorm        metalTensor
	logits            metalTensor

	rawCache        [NLayer]metalTensor
	attnCompCache   [NLayer]metalTensor
	attnStateKV     [NLayer]metalTensor
	attnStateScore  [NLayer]metalTensor
	indexCompCache  [NLayer]metalTensor
	indexStateKV    [NLayer]metalTensor
	indexStateScore [NLayer]metalTensor
	layerNComp      [NLayer]int
	layerNIndexComp [NLayer]int
}

func (e *Engine) initMetalGPU() (interface{}, error) {
	if e.Model == nil || len(e.Model.data) == 0 {
		return nil, fmt.Errorf("Metal: model mmap is empty")
	}
	configureConstrainedMetalEnv()
	if C.ds4_metal_init() == 0 {
		return nil, fmt.Errorf("Metal: ds4_metal_init failed")
	}

	me := &MetalEngine{
		ready:     true,
		strict:    e.StrictGPU,
		enableMoE: os.Getenv("DS4_METAL_ENABLE_MOE") == "1" || e.StrictGPU,
		streaming: true,
		q8MinRows: constrainedMetalEnvInt("DS4_METAL_Q8_MIN_ROWS", 2048),
		model:     e.Model,
		mapPtr:    unsafe.Pointer(&e.Model.data[0]),
		modelSize: uint64(len(e.Model.data)),
	}
	C.ds4_metal_set_model_streaming(C.bool(true))
	if C.ds4_metal_set_model_map(me.mapPtr, C.uint64_t(me.modelSize)) == 0 {
		C.ds4_metal_cleanup()
		return nil, fmt.Errorf("Metal: failed to register model mmap")
	}
	if err := me.configureHotModelRanges(); err != nil {
		C.ds4_metal_cleanup()
		return nil, err
	}
	runtime.KeepAlive(e.Model.data)
	suffix := ""
	if me.enableMoE {
		suffix = " (routed MoE enabled)"
	}
	fmt.Printf("[gpu] Metal backend ready%s\n", suffix)
	return me, nil
}

func (me *MetalEngine) Close() {
	if me == nil {
		return
	}
	me.mu.Lock()
	defer me.mu.Unlock()
	for _, t := range []*metalTensor{
		&me.x, &me.out, &me.moeX, &me.moeGate, &me.moeUp, &me.moeMid,
		&me.moeOut, &me.moeExps, &me.moeSelected, &me.moeWeights,
	} {
		me.freeTensor(t)
	}
	me.freePrefillGraph()
	if me.ready {
		if os.Getenv("DS4_METAL_MEMORY_REPORT") != "" {
			label := C.CString("go metal close")
			C.ds4_metal_print_memory_report(label)
			C.free(unsafe.Pointer(label))
		}
		C.ds4_metal_cleanup()
	}
	me.ready = false
}

func configureConstrainedMetalEnv() {
	setDefaultEnv("DS4_METAL_STREAM_WEIGHTS", "1")
	setDefaultEnv("DS4_METAL_STREAM_RAM_MB", "24576")
	setDefaultEnv("DS4_METAL_RESIDENT_HOT_MB", "12288")
	setDefaultEnv("DS4_METAL_STREAM_CACHE_RAM_MB", "4096")
	setDefaultEnv("DS4_METAL_COMPACT_EXPERT_CACHE_MB", "8192")
	setDefaultEnv("DS4_METAL_STREAM_CACHE", "64")
	setDefaultEnv("DS4_METAL_STREAM_WINDOW_MB", "8")
	setDefaultEnv("DS4_METAL_STREAM_PIN_MAX_MB", "1")
	setDefaultEnv("DS4_METAL_Q8_MIN_ROWS", "0")
	setDefaultEnv("DS4_METAL_MEMORY_REPORT", "1")
	_ = os.Setenv("DS4_METAL_NO_RESIDENCY", "1")

	if mb, ok := envUint("DS4_METAL_STREAM_RAM_MB"); ok && mb > 24576 {
		_ = os.Setenv("DS4_METAL_STREAM_RAM_MB", "24576")
	}
}

func setDefaultEnv(name, value string) {
	if os.Getenv(name) == "" {
		_ = os.Setenv(name, value)
	}
}

func envUint(name string) (uint64, bool) {
	v, err := strconv.ParseUint(os.Getenv(name), 10, 64)
	return v, err == nil
}

func constrainedMetalEnvInt(name string, fallback int) int {
	v, err := strconv.Atoi(os.Getenv(name))
	if err != nil || v < 0 {
		return fallback
	}
	return v
}

type metalHotRange struct {
	offset uint64
	bytes  uint64
}

type metalHotPlan struct {
	ranges       []metalHotRange
	selected     uint64
	skippedCount int
	skippedBytes uint64
}

func (me *MetalEngine) configureHotModelRanges() error {
	plan := me.buildHotModelPlan()
	fmt.Printf("[gpu] Metal hot range plan: %d ranges, %.2f GiB selected, %d skipped (%.2f GiB)\n",
		len(plan.ranges),
		float64(plan.selected)/(1024*1024*1024),
		plan.skippedCount,
		float64(plan.skippedBytes)/(1024*1024*1024))
	if len(plan.ranges) == 0 {
		return nil
	}
	cranges := make([]C.ds4_metal_model_range, len(plan.ranges))
	for i, r := range plan.ranges {
		cranges[i].offset = C.uint64_t(r.offset)
		cranges[i].bytes = C.uint64_t(r.bytes)
	}
	if C.ds4_metal_set_hot_model_ranges(
		me.mapPtr,
		C.uint64_t(me.modelSize),
		&cranges[0],
		C.uint32_t(len(cranges)),
	) == 0 {
		return fmt.Errorf("Metal: failed to configure constrained hot model ranges")
	}
	runtime.KeepAlive(me.model.data)
	return nil
}

func (me *MetalEngine) buildHotModelPlan() metalHotPlan {
	var plan metalHotPlan
	budgetMB, ok := envUint("DS4_METAL_RESIDENT_HOT_MB")
	if !ok || budgetMB == 0 {
		return plan
	}
	budget := budgetMB * 1024 * 1024
	page := uint64(os.Getpagesize())
	add := func(name string) {
		t := me.model.Tensors[name]
		if t == nil || t.DataBytes() == 0 {
			return
		}
		plan.addTensor(t, budget, page)
	}
	for il := 0; il < me.modelLayerCount(); il++ {
		p := fmt.Sprintf("blk.%d.", il)
		for _, suffix := range metalHotLayerTensorSuffixes {
			add(p + suffix)
		}
	}
	for _, name := range metalHotGlobalTensorNames {
		add(name)
	}
	for il := 0; il < me.modelLayerCount(); il++ {
		p := fmt.Sprintf("blk.%d.", il)
		for _, suffix := range metalHotExpertTensorSuffixes {
			add(p + suffix)
		}
	}
	plan.merge()
	return plan
}

func (me *MetalEngine) modelLayerCount() int {
	if me.model != nil {
		if v, ok := me.model.MetaU32("deepseek4.block_count"); ok && v > 0 {
			return int(v)
		}
		if v, ok := me.model.MetaU32("deepseek2.block_count"); ok && v > 0 {
			return int(v)
		}
	}
	return NLayer
}

func (p *metalHotPlan) addTensor(t *GGUFTensor, budget, page uint64) {
	viewOffset := t.AbsOffset & ^(page - 1)
	leading := t.AbsOffset - viewOffset
	viewBytes := roundUpU64(leading+t.DataBytes(), page)
	for _, r := range p.ranges {
		if r.offset == viewOffset && r.bytes == viewBytes {
			return
		}
	}
	if len(p.ranges) >= 2048 || viewBytes > budget {
		p.skippedCount++
		p.skippedBytes += viewBytes
		return
	}
	if p.selected > budget-viewBytes {
		p.merge()
	}
	if p.selected > budget-viewBytes {
		p.skippedCount++
		p.skippedBytes += viewBytes
		return
	}
	p.ranges = append(p.ranges, metalHotRange{offset: viewOffset, bytes: viewBytes})
	p.selected += viewBytes
}

func (p *metalHotPlan) merge() {
	if len(p.ranges) < 2 {
		return
	}
	sort.Slice(p.ranges, func(i, j int) bool {
		if p.ranges[i].offset == p.ranges[j].offset {
			return p.ranges[i].bytes < p.ranges[j].bytes
		}
		return p.ranges[i].offset < p.ranges[j].offset
	})
	out := p.ranges[:0]
	for _, r := range p.ranges {
		if r.bytes == 0 {
			continue
		}
		if len(out) != 0 {
			last := &out[len(out)-1]
			lastEnd := last.offset + last.bytes
			rEnd := r.offset + r.bytes
			if r.offset <= lastEnd {
				if rEnd > lastEnd {
					last.bytes = rEnd - last.offset
				}
				continue
			}
		}
		out = append(out, r)
	}
	var selected uint64
	for _, r := range out {
		selected += r.bytes
	}
	p.ranges = out
	p.selected = selected
}

func roundUpU64(v, a uint64) uint64 {
	if a == 0 {
		return v
	}
	return (v + a - 1) & ^(a - 1)
}

var metalHotGlobalTensorNames = []string{
	"token_embd.weight",
	"output_hc_base.weight",
	"output_hc_fn.weight",
	"output_hc_scale.weight",
	"output_norm.weight",
	"output.weight",
}

var metalHotLayerTensorSuffixes = []string{
	"hc_attn_fn.weight",
	"hc_attn_scale.weight",
	"hc_attn_base.weight",
	"attn_norm.weight",
	"attn_q_a.weight",
	"attn_q_a_norm.weight",
	"attn_q_b.weight",
	"attn_kv.weight",
	"attn_kv_a_norm.weight",
	"attn_sinks.weight",
	"attn_output_a.weight",
	"attn_output_b.weight",
	"attn_compressor_ape.weight",
	"attn_compressor_kv.weight",
	"attn_compressor_gate.weight",
	"attn_compressor_norm.weight",
	"indexer_attn_q_b.weight",
	"indexer_proj.weight",
	"indexer_compressor_ape.weight",
	"indexer_compressor_kv.weight",
	"indexer_compressor_gate.weight",
	"indexer_compressor_norm.weight",
	"hc_ffn_fn.weight",
	"hc_ffn_scale.weight",
	"hc_ffn_base.weight",
	"ffn_norm.weight",
	"ffn_gate_tid2eid.weight",
	"ffn_gate_inp.weight",
	"exp_probs_b.bias",
	"ffn_gate_shexp.weight",
	"ffn_up_shexp.weight",
	"ffn_down_shexp.weight",
	"attn_q.weight",
	"attn_kv_a_mqa.weight",
	"attn_kv_b.weight",
	"attn_output.weight",
}

var metalHotExpertTensorSuffixes = []string{
	"ffn_gate_exps.weight",
	"ffn_up_exps.weight",
	"ffn_down_exps.weight",
}

func (me *MetalEngine) ensureTensor(t *metalTensor, bytes uint64) bool {
	if t.ptr != nil && t.bytes >= bytes {
		return true
	}
	me.freeTensor(t)
	t.ptr = C.ds4_metal_tensor_alloc(C.uint64_t(bytes))
	if t.ptr == nil {
		t.bytes = 0
		return false
	}
	t.bytes = bytes
	return true
}

func (me *MetalEngine) freeTensor(t *metalTensor) {
	if t.ptr != nil {
		C.ds4_metal_tensor_free(t.ptr)
		t.ptr = nil
		t.bytes = 0
	}
}

func (me *MetalEngine) freePrefillGraph() {
	pg := &me.prefill
	for _, t := range []*metalTensor{
		&pg.tokens, &pg.curHC, &pg.nextHC, &pg.flatHC, &pg.hcMix, &pg.hcSplit,
		&pg.attnCur, &pg.attnNorm, &pg.qr, &pg.qrNorm, &pg.q, &pg.kvRaw, &pg.kv,
		&pg.compKV, &pg.compSC, &pg.indexerQ, &pg.indexerWeights, &pg.indexerScores,
		&pg.topK, &pg.heads, &pg.attnLow, &pg.attnOut, &pg.groupTmp, &pg.lowTmp,
		&pg.afterAttnHC, &pg.ffnCur, &pg.ffnNorm, &pg.sharedGate, &pg.sharedUp,
		&pg.sharedMid, &pg.sharedOut, &pg.routerLogits, &pg.routerProbs,
		&pg.routerSelected, &pg.routerWeights, &pg.routedGate, &pg.routedUp,
		&pg.routedMid, &pg.routedDown, &pg.routedOut, &pg.outputPre,
		&pg.outputWeights, &pg.outputEmbd, &pg.outputNorm, &pg.logits,
	} {
		me.freeTensor(t)
	}
	for i := 0; i < NLayer; i++ {
		me.freeTensor(&pg.rawCache[i])
		me.freeTensor(&pg.attnCompCache[i])
		me.freeTensor(&pg.attnStateKV[i])
		me.freeTensor(&pg.attnStateScore[i])
		me.freeTensor(&pg.indexCompCache[i])
		me.freeTensor(&pg.indexStateKV[i])
		me.freeTensor(&pg.indexStateScore[i])
	}
	*pg = metalPrefillGraph{}
}

func (me *MetalEngine) ensurePrefillGraph(s *Session, nTokens int) bool {
	if s == nil || s.Engine == nil || s.Engine.Config == nil || nTokens <= 0 || nTokens > s.CtxSize {
		return false
	}
	cfg := s.Engine.Config
	if cfg.NHC != NHC || cfg.NLayer != NLayer || cfg.NEmbd != NEmbd || cfg.NHeadDim != NHeadDim ||
		cfg.NHead != NHead || cfg.NVocab != NVocab || cfg.NExpert != NExpert || cfg.NExpertUsed != NExpertUsed {
		return false
	}
	rawCap := nTokens
	if rawCap < cfg.NSWA {
		rawCap = cfg.NSWA
	}
	if rawCap > s.CtxSize {
		rawCap = s.CtxSize
	}
	compCap := s.CtxSize/4 + 2
	if compCap < 2 {
		compCap = 2
	}
	pg := &me.prefill
	if pg.allocated && pg.ctxSize == s.CtxSize && pg.capTokens == nTokens && pg.rawCap == rawCap && pg.compCap == compCap {
		return true
	}
	me.freePrefillGraph()
	pg = &me.prefill
	pg.ctxSize = s.CtxSize
	pg.capTokens = nTokens
	pg.rawCap = rawCap
	pg.compCap = compCap

	hcDim := uint64(cfg.NHC * cfg.NEmbd)
	mixHC := uint64(2*cfg.NHC + cfg.NHC*cfg.NHC)
	qRank := uint64(cfg.NLoraQ)
	qDim := uint64(cfg.NHead * cfg.NHeadDim)
	lowDim := uint64(cfg.NOutGroup * cfg.NLoraO)
	groupDim := uint64(cfg.NHeadDim * (cfg.NHead / cfg.NOutGroup))
	indexerQDim := uint64(cfg.NIndexerHead * cfg.NIndexerHeadDim)
	sharedDim := uint64(cfg.NFFExp)
	routedMidDim := uint64(cfg.NFFExp)
	pc := uint64(nTokens)
	f4 := uint64(4)

	alloc := func(t *metalTensor, elems uint64, elemBytes uint64) bool {
		return me.ensureTensor(t, elems*elemBytes)
	}
	ok := alloc(&pg.tokens, pc, 4) &&
		alloc(&pg.curHC, pc*hcDim, f4) && alloc(&pg.nextHC, pc*hcDim, f4) &&
		alloc(&pg.flatHC, pc*hcDim, f4) && alloc(&pg.hcMix, pc*mixHC, f4) &&
		alloc(&pg.hcSplit, pc*mixHC, f4) && alloc(&pg.attnCur, pc*uint64(cfg.NEmbd), f4) &&
		alloc(&pg.attnNorm, pc*uint64(cfg.NEmbd), f4) && alloc(&pg.qr, pc*qRank, f4) &&
		alloc(&pg.qrNorm, pc*qRank, f4) && alloc(&pg.q, pc*qDim, f4) &&
		alloc(&pg.kvRaw, pc*uint64(cfg.NHeadDim), f4) && alloc(&pg.kv, pc*uint64(cfg.NHeadDim), f4) &&
		alloc(&pg.compKV, pc*uint64(2*cfg.NHeadDim), f4) && alloc(&pg.compSC, pc*uint64(2*cfg.NHeadDim), f4) &&
		alloc(&pg.indexerQ, pc*indexerQDim, f4) && alloc(&pg.indexerWeights, pc*uint64(cfg.NIndexerHead), f4) &&
		alloc(&pg.indexerScores, uint64(compCap)*pc, f4) && alloc(&pg.topK, uint64(cfg.NIndexerTopK)*pc, 4) &&
		alloc(&pg.heads, pc*qDim, f4) && alloc(&pg.attnLow, pc*lowDim, f4) &&
		alloc(&pg.attnOut, pc*uint64(cfg.NEmbd), f4) && alloc(&pg.groupTmp, pc*groupDim, f4) &&
		alloc(&pg.lowTmp, pc*uint64(cfg.NLoraO), f4) && alloc(&pg.afterAttnHC, pc*hcDim, f4) &&
		alloc(&pg.ffnCur, pc*uint64(cfg.NEmbd), f4) && alloc(&pg.ffnNorm, pc*uint64(cfg.NEmbd), f4) &&
		alloc(&pg.sharedGate, pc*sharedDim, f4) && alloc(&pg.sharedUp, pc*sharedDim, f4) &&
		alloc(&pg.sharedMid, pc*sharedDim, f4) && alloc(&pg.sharedOut, pc*uint64(cfg.NEmbd), f4) &&
		alloc(&pg.routerLogits, pc*uint64(cfg.NExpert), f4) && alloc(&pg.routerProbs, pc*uint64(cfg.NExpert), f4) &&
		alloc(&pg.routerSelected, pc*uint64(cfg.NExpertUsed), 4) && alloc(&pg.routerWeights, pc*uint64(cfg.NExpertUsed), f4) &&
		alloc(&pg.routedGate, pc*uint64(cfg.NExpertUsed)*routedMidDim, f4) &&
		alloc(&pg.routedUp, pc*uint64(cfg.NExpertUsed)*routedMidDim, f4) &&
		alloc(&pg.routedMid, pc*uint64(cfg.NExpertUsed)*routedMidDim, f4) &&
		alloc(&pg.routedDown, pc*uint64(cfg.NExpertUsed)*uint64(cfg.NEmbd), f4) &&
		alloc(&pg.routedOut, pc*uint64(cfg.NEmbd), f4) &&
		alloc(&pg.outputPre, uint64(cfg.NHC), f4) && alloc(&pg.outputWeights, uint64(cfg.NHC), f4) &&
		alloc(&pg.outputEmbd, uint64(cfg.NEmbd), f4) && alloc(&pg.outputNorm, uint64(cfg.NEmbd), f4) &&
		alloc(&pg.logits, uint64(cfg.NVocab), f4)
	for il := 0; ok && il < cfg.NLayer; il++ {
		ratio := layerCompressRatio(il)
		ok = alloc(&pg.rawCache[il], uint64(rawCap*cfg.NHeadDim), f4)
		if !ok || ratio == 0 {
			continue
		}
		coff := 1
		if ratio == 4 {
			coff = 2
		}
		attnWidth := coff * cfg.NHeadDim
		attnRows := coff * ratio
		ok = alloc(&pg.attnCompCache[il], uint64(compCap*cfg.NHeadDim), f4) &&
			alloc(&pg.attnStateKV[il], uint64(attnWidth*attnRows), f4) &&
			alloc(&pg.attnStateScore[il], uint64(attnWidth*attnRows), f4) &&
			me.fillTensorF32(&pg.attnStateKV[il], 0, attnWidth*attnRows) &&
			me.fillTensorF32(&pg.attnStateScore[il], -1.0e30, attnWidth*attnRows)
		if ok && ratio == 4 {
			indexWidth := coff * cfg.NIndexerHeadDim
			indexRows := coff * ratio
			ok = alloc(&pg.indexCompCache[il], uint64(compCap*cfg.NIndexerHeadDim), f4) &&
				alloc(&pg.indexStateKV[il], uint64(indexWidth*indexRows), f4) &&
				alloc(&pg.indexStateScore[il], uint64(indexWidth*indexRows), f4) &&
				me.fillTensorF32(&pg.indexStateKV[il], 0, indexWidth*indexRows) &&
				me.fillTensorF32(&pg.indexStateScore[il], -1.0e30, indexWidth*indexRows)
		}
	}
	if !ok {
		me.freePrefillGraph()
		return false
	}
	pg.allocated = true
	return true
}

func (me *MetalEngine) fillTensorF32(t *metalTensor, v float32, n int) bool {
	if t.ptr == nil || n < 0 || t.bytes < uint64(n*4) {
		return false
	}
	p := C.ds4_metal_tensor_contents(t.ptr)
	if p == nil {
		return false
	}
	buf := unsafe.Slice((*float32)(p), n)
	for i := range buf {
		buf[i] = v
	}
	return true
}

func metalTensorView(base *metalTensor, offsetBytes, bytes uint64) metalTensor {
	if base == nil || base.ptr == nil || bytes == 0 || offsetBytes > base.bytes || bytes > base.bytes-offsetBytes {
		return metalTensor{}
	}
	ptr := C.ds4_metal_tensor_view(base.ptr, C.uint64_t(offsetBytes), C.uint64_t(bytes))
	if ptr == nil {
		return metalTensor{}
	}
	return metalTensor{ptr: ptr, bytes: bytes}
}

func freeMetalView(t metalTensor) {
	if t.ptr != nil {
		C.ds4_metal_tensor_free(t.ptr)
	}
}

func (me *MetalEngine) tensor(name string) *GGUFTensor {
	if me == nil || me.model == nil {
		return nil
	}
	return me.model.Tensors[name]
}

func (me *MetalEngine) layerTensor(il int, suffix string) *GGUFTensor {
	return me.tensor(fmt.Sprintf("blk.%d.%s", il, suffix))
}

func tensorOffset(t *GGUFTensor) C.uint64_t {
	if t == nil {
		return 0
	}
	return C.uint64_t(t.AbsOffset)
}

func tensorType(t *GGUFTensor) C.uint32_t {
	if t == nil {
		return 0
	}
	return C.uint32_t(t.Type)
}

func ropeParams(cfg *ModelConfig, il int) (freqBase, freqScale, extFactor, attnFactor float32, origCtx uint32) {
	ratio := layerCompressRatio(il)
	if ratio != 0 {
		freqBase = cfg.CompressRoPEFreqBase
		if freqBase == 0 {
			freqBase = cfg.RoPEFreqBase
		}
		if cfg.RoPEScaleFactor > 0 {
			freqScale = 1.0 / cfg.RoPEScaleFactor
		} else {
			freqScale = 1.0
		}
		if cfg.RoPEScaleFactor > 1.0 {
			extFactor = 1.0
		}
		origCtx = uint32(cfg.RoPEOrigCtx)
	} else {
		freqBase = cfg.RoPEFreqBase
		freqScale = 1.0
	}
	attnFactor = 1.0
	if extFactor != 0 && freqScale > 0 {
		attnFactor /= 1.0 + 0.1*float32(math.Log(1.0/float64(freqScale)))
	}
	return
}

func (me *MetalEngine) prefillGraph(s *Session, tokens []int) bool {
	if me == nil || !me.ready || s == nil || s.Pos != 0 || len(tokens) == 0 {
		return false
	}
	cfg := s.Engine.Config
	if cfg == nil || cfg.NHC == 0 || len(tokens) > s.CtxSize {
		return false
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= cfg.NVocab {
			return false
		}
	}

	me.mu.Lock()
	defer me.mu.Unlock()
	if !me.ready || !me.ensurePrefillGraph(s, len(tokens)) {
		return false
	}
	pg := &me.prefill
	for il := 0; il < cfg.NLayer; il++ {
		pg.layerNComp[il] = 0
		pg.layerNIndexComp[il] = 0
	}

	token32 := make([]int32, len(tokens))
	for i, tok := range tokens {
		token32[i] = int32(tok)
	}
	if C.ds4_metal_tensor_write(pg.tokens.ptr, 0, unsafe.Pointer(&token32[0]), C.uint64_t(len(token32)*4)) == 0 {
		return false
	}

	if C.ds4_metal_begin_commands() == 0 {
		return false
	}
	ok := C.ds4_metal_embed_tokens_hc_tensor(
		pg.curHC.ptr,
		pg.tokens.ptr,
		me.mapPtr,
		C.uint64_t(me.modelSize),
		tensorOffset(me.tensor("token_embd.weight")),
		C.uint32_t(cfg.NVocab),
		C.uint32_t(len(tokens)),
		C.uint32_t(cfg.NEmbd),
		C.uint32_t(cfg.NHC),
	) != 0
	ok = me.finishCommandBatch(ok)
	if !ok {
		return false
	}

	for il := 0; il < cfg.NLayer; il++ {
		if C.ds4_metal_begin_commands() == 0 {
			return false
		}
		ok = me.prefillLayerAttention(s, il, len(tokens))
		ok = me.finishCommandBatch(ok)
		if !ok {
			return false
		}
		if C.ds4_metal_begin_commands() == 0 {
			return false
		}
		ok = me.prefillLayerFFN(s, il, len(tokens))
		ok = me.finishCommandBatch(ok)
		if !ok {
			return false
		}
		pg.curHC, pg.nextHC = pg.nextHC, pg.curHC
	}

	if C.ds4_metal_begin_commands() == 0 {
		return false
	}
	ok = me.prefillOutputHead(s, len(tokens))
	ok = me.finishCommandBatch(ok)
	if !ok {
		return false
	}
	if C.ds4_metal_tensor_read(pg.logits.ptr, 0, unsafe.Pointer(&s.Logits[0]), C.uint64_t(cfg.NVocab*4)) == 0 {
		return false
	}
	lastOff := uint64((len(tokens) - 1) * cfg.NHC * cfg.NEmbd * 4)
	if C.ds4_metal_tensor_read(pg.curHC.ptr, C.uint64_t(lastOff), unsafe.Pointer(&s.Decode.CurHC[0]), C.uint64_t(cfg.NHC*cfg.NEmbd*4)) == 0 {
		return false
	}
	if !me.readPrefillCaches(s, len(tokens)) {
		return false
	}
	s.Tokens = append(s.Tokens, tokens...)
	s.Pos += len(tokens)
	runtime.KeepAlive(me.model.data)
	return true
}

func (me *MetalEngine) finishCommandBatch(ok bool) bool {
	if ok {
		return C.ds4_metal_end_commands() != 0
	}
	C.ds4_metal_end_commands()
	C.ds4_metal_synchronize()
	return false
}

func (me *MetalEngine) prefillLayerAttention(s *Session, il, nTokens int) bool {
	cfg := s.Engine.Config
	pg := &me.prefill
	hcDim := uint64(cfg.NHC * cfg.NEmbd)
	mixHC := uint64(2*cfg.NHC + cfg.NHC*cfg.NHC)
	qRank := uint64(cfg.NLoraQ)
	qDim := uint64(cfg.NHead * cfg.NHeadDim)
	groupHeads := cfg.NHead / cfg.NOutGroup
	groupDim := uint64(cfg.NHeadDim * groupHeads)
	ratio := layerCompressRatio(il)
	compressed := ratio != 0
	freqBase, freqScale, extFactor, attnFactor, origCtx := ropeParams(cfg, il)

	hcMix := metalTensorView(&pg.hcMix, 0, uint64(nTokens)*mixHC*4)
	hcSplit := metalTensorView(&pg.hcSplit, 0, uint64(nTokens)*mixHC*4)
	attnCur := metalTensorView(&pg.attnCur, 0, uint64(nTokens*cfg.NEmbd*4))
	afterAttn := metalTensorView(&pg.afterAttnHC, 0, uint64(nTokens)*hcDim*4)
	defer freeMetalView(hcMix)
	defer freeMetalView(hcSplit)
	defer freeMetalView(attnCur)
	defer freeMetalView(afterAttn)
	if hcMix.ptr == nil || hcSplit.ptr == nil || attnCur.ptr == nil || afterAttn.ptr == nil {
		return false
	}
	ok := C.ds4_metal_rms_norm_plain_rows_tensor(pg.flatHC.ptr, pg.curHC.ptr, C.uint32_t(hcDim), C.uint32_t(nTokens), C.float(cfg.RMSEps)) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(hcMix.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "hc_attn_fn.weight")), C.uint64_t(hcDim), C.uint64_t(mixHC), pg.flatHC.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_hc_split_weighted_sum_tensor(attnCur.ptr, hcSplit.ptr, hcMix.ptr, pg.curHC.ptr,
			me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "hc_attn_scale.weight")),
			tensorOffset(me.layerTensor(il, "hc_attn_base.weight")),
			C.uint32_t(cfg.NEmbd), C.uint32_t(cfg.NHC), C.uint32_t(cfg.NHCSinkhornIter), C.float(cfg.HCEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_rms_norm_weight_rows_tensor(pg.attnNorm.ptr, attnCur.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_norm.weight")), C.uint32_t(cfg.NEmbd), C.uint32_t(nTokens), C.float(cfg.RMSEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_q8_0_tensor(pg.qr.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_q_a.weight")), C.uint64_t(cfg.NEmbd), C.uint64_t(cfg.NLoraQ), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_q8_0_tensor(pg.kvRaw.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_kv.weight")), C.uint64_t(cfg.NEmbd), C.uint64_t(cfg.NHeadDim), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_dsv4_qkv_rms_norm_rows_tensor(pg.qrNorm.ptr, pg.qr.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_q_a_norm.weight")), C.uint32_t(cfg.NLoraQ),
			pg.kv.ptr, pg.kvRaw.ptr, tensorOffset(me.layerTensor(il, "attn_kv_a_norm.weight")),
			C.uint32_t(cfg.NHeadDim), C.uint32_t(nTokens), C.float(cfg.RMSEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_q8_0_tensor(pg.q.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_q_b.weight")), C.uint64_t(cfg.NLoraQ), C.uint64_t(cfg.NHead*cfg.NHeadDim), pg.qrNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_head_rms_norm_tensor(pg.q.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim), C.float(cfg.RMSEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_rope_tail_tensor(pg.q.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim),
			C.uint32_t(cfg.NRot), 0, C.uint32_t(origCtx), C.bool(false), C.float(freqBase), C.float(freqScale),
			C.float(extFactor), C.float(attnFactor), C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow)) != 0
	}
	if ok {
		ok = C.ds4_metal_rope_tail_tensor(pg.kv.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NHeadKV), C.uint32_t(cfg.NHeadDim),
			C.uint32_t(cfg.NRot), 0, C.uint32_t(origCtx), C.bool(false), C.float(freqBase), C.float(freqScale),
			C.float(extFactor), C.float(attnFactor), C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow)) != 0
	}
	if ok {
		ok = C.ds4_metal_dsv4_fp8_kv_quantize_tensor(pg.kv.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NHeadDim), C.uint32_t(cfg.NRot)) != 0
	}
	if ok {
		ok = C.ds4_metal_store_raw_kv_batch_tensor(pg.rawCache[il].ptr, pg.kv.ptr, C.uint32_t(pg.rawCap), 0, C.uint32_t(nTokens), C.uint32_t(cfg.NHeadDim)) != 0
	}

	nComp := 0
	if ok && compressed {
		nComp = nTokens / ratio
		ok = me.prefillCompressedAttentionState(s, il, ratio, nTokens, freqBase, freqScale, extFactor, attnFactor, origCtx)
	}
	if ok {
		switch {
		case ratio == 0:
			ok = C.ds4_metal_attention_prefill_raw_heads_tensor(pg.heads.ptr, me.mapPtr, C.uint64_t(me.modelSize),
				tensorOffset(me.layerTensor(il, "attn_sinks.weight")), pg.q.ptr, pg.kv.ptr, C.uint32_t(nTokens),
				C.uint32_t(cfg.NSWA), C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim)) != 0
		case nComp == 0:
			ok = C.ds4_metal_attention_prefill_raw_heads_tensor(pg.heads.ptr, me.mapPtr, C.uint64_t(me.modelSize),
				tensorOffset(me.layerTensor(il, "attn_sinks.weight")), pg.q.ptr, pg.kv.ptr, C.uint32_t(nTokens),
				C.uint32_t(cfg.NSWA), C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim)) != 0
		case ratio == 4 && nComp > cfg.NIndexerTopK:
			scale := float32(1.0 / math.Sqrt(float64(cfg.NIndexerHeadDim*cfg.NIndexerHead)))
			ok = C.ds4_metal_indexer_scores_prefill_tensor(pg.indexerScores.ptr, pg.indexerQ.ptr, pg.indexerWeights.ptr,
				pg.indexCompCache[il].ptr, C.uint32_t(nComp), C.uint32_t(nTokens),
				C.uint32_t(cfg.NIndexerHead), C.uint32_t(cfg.NIndexerHeadDim), C.uint32_t(ratio), C.float(scale)) != 0
			if ok {
				ok = C.ds4_metal_indexer_topk_tensor(pg.topK.ptr, pg.indexerScores.ptr, C.uint32_t(nComp), C.uint32_t(nTokens), C.uint32_t(cfg.NIndexerTopK)) != 0
			}
			if ok {
				ok = C.ds4_metal_attention_indexed_mixed_batch_heads_tensor(pg.heads.ptr, me.mapPtr, C.uint64_t(me.modelSize),
					tensorOffset(me.layerTensor(il, "attn_sinks.weight")), pg.q.ptr, pg.rawCache[il].ptr, pg.attnCompCache[il].ptr, pg.topK.ptr,
					C.uint32_t(nTokens), 0, C.uint32_t(nTokens), C.uint32_t(pg.rawCap), 0,
					C.uint32_t(nComp), C.uint32_t(cfg.NIndexerTopK), C.uint32_t(cfg.NSWA), C.uint32_t(ratio),
					C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim)) != 0
			}
		default:
			ok = C.ds4_metal_attention_prefill_static_mixed_heads_tensor(pg.heads.ptr, me.mapPtr, C.uint64_t(me.modelSize),
				tensorOffset(me.layerTensor(il, "attn_sinks.weight")), pg.q.ptr, pg.kv.ptr, pg.attnCompCache[il].ptr,
				C.uint32_t(nTokens), C.uint32_t(nComp), C.uint32_t(cfg.NSWA), C.uint32_t(ratio),
				C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim)) != 0
		}
	}
	if ok {
		ok = C.ds4_metal_rope_tail_tensor(pg.heads.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NHead), C.uint32_t(cfg.NHeadDim),
			C.uint32_t(cfg.NRot), 0, C.uint32_t(origCtx), C.bool(true), C.float(freqBase), C.float(freqScale),
			C.float(extFactor), C.float(attnFactor), C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow)) != 0
	}
	if ok {
		ok = C.ds4_metal_attention_output_q8_batch_tensor(pg.attnOut.ptr, pg.attnLow.ptr, pg.groupTmp.ptr, pg.lowTmp.ptr,
			me.mapPtr, C.uint64_t(me.modelSize), tensorOffset(me.layerTensor(il, "attn_output_a.weight")),
			tensorOffset(me.layerTensor(il, "attn_output_b.weight")), C.uint64_t(groupDim), C.uint64_t(cfg.NLoraO),
			C.uint32_t(cfg.NOutGroup), C.uint64_t(cfg.NEmbd), pg.heads.ptr, C.uint32_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_hc_expand_split_tensor(afterAttn.ptr, pg.attnOut.ptr, pg.curHC.ptr, hcSplit.ptr,
			C.uint32_t(cfg.NEmbd), C.uint32_t(cfg.NHC)) != 0
	}
	_ = qRank
	_ = qDim
	return ok
}

func (me *MetalEngine) prefillCompressedAttentionState(
	s *Session, il, ratio, nTokens int,
	freqBase, freqScale, extFactor, attnFactor float32,
	origCtx uint32,
) bool {
	cfg := s.Engine.Config
	pg := &me.prefill
	coff := 1
	if ratio == 4 {
		coff = 2
	}
	compWidth := coff * cfg.NHeadDim
	nComp := nTokens / ratio
	if nComp > pg.compCap {
		return false
	}
	required := []string{
		"attn_compressor_kv.weight",
		"attn_compressor_gate.weight",
		"attn_compressor_ape.weight",
		"attn_compressor_norm.weight",
	}
	for _, suffix := range required {
		if me.layerTensor(il, suffix) == nil {
			return false
		}
	}
	ok := C.ds4_metal_matmul_f16_tensor(pg.compKV.ptr, me.mapPtr, C.uint64_t(me.modelSize),
		tensorOffset(me.layerTensor(il, "attn_compressor_kv.weight")),
		C.uint64_t(cfg.NEmbd), C.uint64_t(compWidth), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.compSC.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_compressor_gate.weight")),
			C.uint64_t(cfg.NEmbd), C.uint64_t(compWidth), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_compressor_prefill_tensor(pg.attnCompCache[il].ptr, pg.attnStateKV[il].ptr, pg.attnStateScore[il].ptr,
			pg.compKV.ptr, pg.compSC.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "attn_compressor_ape.weight")), tensorType(me.layerTensor(il, "attn_compressor_ape.weight")),
			tensorOffset(me.layerTensor(il, "attn_compressor_norm.weight")), tensorType(me.layerTensor(il, "attn_compressor_norm.weight")),
			C.uint32_t(cfg.NHeadDim), C.uint32_t(ratio), 0, C.uint32_t(nTokens), C.uint32_t(cfg.NRot), C.uint32_t(origCtx),
			C.bool(true), C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
			C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow), C.float(cfg.RMSEps)) != 0
	}
	if ok && ratio == 4 && nTokens >= 4 {
		ok = me.refreshRatio4State(s, il, pg.attnStateKV[il].ptr, pg.attnStateScore[il].ptr,
			"attn_compressor_kv.weight", "attn_compressor_gate.weight", "attn_compressor_ape.weight",
			cfg.NHeadDim, compWidth, nTokens)
	}
	if !ok {
		return false
	}
	pg.layerNComp[il] = nComp

	if ratio != 4 {
		return true
	}
	required = []string{
		"indexer_compressor_kv.weight",
		"indexer_compressor_gate.weight",
		"indexer_compressor_ape.weight",
		"indexer_compressor_norm.weight",
		"indexer_attn_q_b.weight",
		"indexer_proj.weight",
	}
	for _, suffix := range required {
		if me.layerTensor(il, suffix) == nil {
			return false
		}
	}
	indexWidth := coff * cfg.NIndexerHeadDim
	ok = C.ds4_metal_matmul_f16_tensor(pg.compKV.ptr, me.mapPtr, C.uint64_t(me.modelSize),
		tensorOffset(me.layerTensor(il, "indexer_compressor_kv.weight")),
		C.uint64_t(cfg.NEmbd), C.uint64_t(indexWidth), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.compSC.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "indexer_compressor_gate.weight")),
			C.uint64_t(cfg.NEmbd), C.uint64_t(indexWidth), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.indexerQ.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "indexer_attn_q_b.weight")),
			C.uint64_t(cfg.NLoraQ), C.uint64_t(cfg.NIndexerHead*cfg.NIndexerHeadDim), pg.qrNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_rope_tail_tensor(pg.indexerQ.ptr, C.uint32_t(nTokens), C.uint32_t(cfg.NIndexerHead), C.uint32_t(cfg.NIndexerHeadDim),
			C.uint32_t(cfg.NRot), 0, C.uint32_t(origCtx), C.bool(false), C.float(freqBase), C.float(freqScale),
			C.float(extFactor), C.float(attnFactor), C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.indexerWeights.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "indexer_proj.weight")),
			C.uint64_t(cfg.NEmbd), C.uint64_t(cfg.NIndexerHead), pg.attnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_compressor_prefill_tensor(pg.indexCompCache[il].ptr, pg.indexStateKV[il].ptr, pg.indexStateScore[il].ptr,
			pg.compKV.ptr, pg.compSC.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "indexer_compressor_ape.weight")), tensorType(me.layerTensor(il, "indexer_compressor_ape.weight")),
			tensorOffset(me.layerTensor(il, "indexer_compressor_norm.weight")), tensorType(me.layerTensor(il, "indexer_compressor_norm.weight")),
			C.uint32_t(cfg.NIndexerHeadDim), C.uint32_t(ratio), 0, C.uint32_t(nTokens), C.uint32_t(cfg.NRot), C.uint32_t(origCtx),
			C.bool(false), C.float(freqBase), C.float(freqScale), C.float(extFactor), C.float(attnFactor),
			C.float(cfg.RoPEYarnBetaFast), C.float(cfg.RoPEYarnBetaSlow), C.float(cfg.RMSEps)) != 0
	}
	if ok && nTokens >= 4 {
		ok = me.refreshRatio4State(s, il, pg.indexStateKV[il].ptr, pg.indexStateScore[il].ptr,
			"indexer_compressor_kv.weight", "indexer_compressor_gate.weight", "indexer_compressor_ape.weight",
			cfg.NIndexerHeadDim, indexWidth, nTokens)
	}
	if ok {
		pg.layerNIndexComp[il] = nComp
	}
	return ok
}

func (me *MetalEngine) refreshRatio4State(
	s *Session, il int,
	stateKV, stateScore *C.ds4_metal_tensor,
	kvSuffix, scoreSuffix, apeSuffix string,
	headDim, width, nTokens int,
) bool {
	if nTokens < 4 || headDim == 0 || width == 0 {
		return true
	}
	cfg := s.Engine.Config
	pg := &me.prefill
	tail := metalTensorView(&pg.attnNorm, uint64((nTokens-4)*cfg.NEmbd*4), uint64(4*cfg.NEmbd*4))
	defer freeMetalView(tail)
	if tail.ptr == nil {
		return false
	}
	ok := C.ds4_metal_matmul_f16_tensor(pg.compKV.ptr, me.mapPtr, C.uint64_t(me.modelSize),
		tensorOffset(me.layerTensor(il, kvSuffix)), C.uint64_t(cfg.NEmbd), C.uint64_t(width), tail.ptr, 4) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.compSC.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, scoreSuffix)), C.uint64_t(cfg.NEmbd), C.uint64_t(width), tail.ptr, 4) != 0
	}
	if ok {
		ok = C.ds4_metal_compressor_prefill_state_ratio4_tensor(stateKV, stateScore, pg.compKV.ptr, pg.compSC.ptr,
			me.mapPtr, C.uint64_t(me.modelSize), tensorOffset(me.layerTensor(il, apeSuffix)), tensorType(me.layerTensor(il, apeSuffix)),
			C.uint32_t(headDim), C.uint32_t(nTokens-4)) != 0
	}
	return ok
}

func (me *MetalEngine) prefillLayerFFN(s *Session, il, nTokens int) bool {
	cfg := s.Engine.Config
	pg := &me.prefill
	hcDim := uint64(cfg.NHC * cfg.NEmbd)
	mixHC := uint64(2*cfg.NHC + cfg.NHC*cfg.NHC)
	hcMix := metalTensorView(&pg.hcMix, 0, uint64(nTokens)*mixHC*4)
	hcSplit := metalTensorView(&pg.hcSplit, 0, uint64(nTokens)*mixHC*4)
	ffnCur := metalTensorView(&pg.ffnCur, 0, uint64(nTokens*cfg.NEmbd*4))
	nextHC := metalTensorView(&pg.nextHC, 0, uint64(nTokens)*hcDim*4)
	defer freeMetalView(hcMix)
	defer freeMetalView(hcSplit)
	defer freeMetalView(ffnCur)
	defer freeMetalView(nextHC)
	if hcMix.ptr == nil || hcSplit.ptr == nil || ffnCur.ptr == nil || nextHC.ptr == nil {
		return false
	}

	ok := C.ds4_metal_rms_norm_plain_rows_tensor(pg.flatHC.ptr, pg.afterAttnHC.ptr, C.uint32_t(hcDim), C.uint32_t(nTokens), C.float(cfg.RMSEps)) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(hcMix.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "hc_ffn_fn.weight")), C.uint64_t(hcDim), C.uint64_t(mixHC), pg.flatHC.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_hc_split_weighted_sum_tensor(ffnCur.ptr, hcSplit.ptr, hcMix.ptr, pg.afterAttnHC.ptr,
			me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "hc_ffn_scale.weight")),
			tensorOffset(me.layerTensor(il, "hc_ffn_base.weight")),
			C.uint32_t(cfg.NEmbd), C.uint32_t(cfg.NHC), C.uint32_t(cfg.NHCSinkhornIter), C.float(cfg.HCEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_rms_norm_weight_rows_tensor(pg.ffnNorm.ptr, ffnCur.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "ffn_norm.weight")), C.uint32_t(cfg.NEmbd), C.uint32_t(nTokens), C.float(cfg.RMSEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.routerLogits.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "ffn_gate_inp.weight")),
			C.uint64_t(cfg.NEmbd), C.uint64_t(cfg.NExpert), pg.ffnNorm.ptr, C.uint64_t(nTokens)) != 0
	}
	hashT := me.layerTensor(il, "ffn_gate_tid2eid.weight")
	hashRows := uint32(0)
	if hashT != nil && hashT.NDim > 1 {
		hashRows = uint32(hashT.Dims[1])
	}
	nRouted := cfg.NExpertUsed
	if s.Engine.FastExperts && nRouted > 2 {
		nRouted -= 2
	}
	biasT := me.layerTensor(il, "exp_probs_b.bias")
	if ok {
		ok = C.ds4_metal_router_select_batch_tensor(pg.routerSelected.ptr, pg.routerWeights.ptr, pg.routerProbs.ptr,
			me.mapPtr, C.uint64_t(me.modelSize), tensorOffset(biasT), tensorOffset(hashT), C.uint32_t(hashRows),
			0, 0, C.bool(biasT != nil), C.bool(hashT != nil), pg.routerLogits.ptr, pg.tokens.ptr, C.uint32_t(nTokens), C.uint32_t(nRouted)) != 0
	}

	layer := &s.Engine.Weights.Layer[il]
	ed := detectExpertDims(layer, cfg)
	gateT := me.layerTensor(il, "ffn_gate_exps.weight")
	upT := me.layerTensor(il, "ffn_up_exps.weight")
	downT := me.layerTensor(il, "ffn_down_exps.weight")
	if ok && (gateT == nil || upT == nil || downT == nil || ed.outDim <= 0 || ed.gateRowBytes <= 0 || ed.downRowBytes <= 0) {
		ok = false
	}
	if ok {
		gateExpertBytes := uint64(ed.outDim * ed.gateRowBytes)
		downExpertBytes := uint64(cfg.NEmbd * ed.downRowBytes)
		ok = C.ds4_metal_routed_moe_batch_tensor(pg.routedOut.ptr, pg.routedGate.ptr, pg.routedUp.ptr, pg.routedMid.ptr, pg.routedDown.ptr,
			me.mapPtr, C.uint64_t(me.modelSize), tensorOffset(gateT), tensorOffset(upT), tensorOffset(downT),
			tensorType(gateT), tensorType(downT), C.uint64_t(gateExpertBytes), C.uint64_t(ed.gateRowBytes),
			C.uint64_t(downExpertBytes), C.uint64_t(ed.downRowBytes), C.uint32_t(cfg.NEmbd), C.uint32_t(ed.outDim),
			C.uint32_t(cfg.NEmbd), pg.routerSelected.ptr, pg.routerWeights.ptr, C.uint32_t(nRouted),
			C.float(cfg.SwiGLUClampExp), pg.ffnNorm.ptr, C.uint32_t(nTokens)) != 0
	}
	sharedDim := detectOutDim(layer.FfnGateShexp, cfg.NEmbd)
	if sharedDim <= 0 {
		sharedDim = cfg.NFFExp
	}
	if ok {
		ok = C.ds4_metal_shared_gate_up_swiglu_q8_0_batch_tensor(pg.sharedGate.ptr, pg.sharedUp.ptr, pg.sharedMid.ptr,
			me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "ffn_gate_shexp.weight")),
			tensorOffset(me.layerTensor(il, "ffn_up_shexp.weight")),
			C.uint64_t(cfg.NEmbd), C.uint64_t(sharedDim), pg.ffnNorm.ptr, C.uint32_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_q8_0_tensor(pg.sharedOut.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.layerTensor(il, "ffn_down_shexp.weight")), C.uint64_t(sharedDim), C.uint64_t(cfg.NEmbd), pg.sharedMid.ptr, C.uint64_t(nTokens)) != 0
	}
	if ok {
		ok = C.ds4_metal_hc_expand_add_split_tensor(nextHC.ptr, pg.routedOut.ptr, pg.sharedOut.ptr, pg.afterAttnHC.ptr, hcSplit.ptr,
			C.uint32_t(cfg.NEmbd), C.uint32_t(cfg.NHC)) != 0
	}
	return ok
}

func (me *MetalEngine) prefillOutputHead(s *Session, nTokens int) bool {
	cfg := s.Engine.Config
	pg := &me.prefill
	hcDim := uint64(cfg.NHC * cfg.NEmbd)
	lastHC := metalTensorView(&pg.curHC, uint64((nTokens-1)*cfg.NHC*cfg.NEmbd*4), hcDim*4)
	defer freeMetalView(lastHC)
	if lastHC.ptr == nil {
		return false
	}
	ok := C.ds4_metal_rms_norm_plain_tensor(pg.flatHC.ptr, lastHC.ptr, C.uint32_t(hcDim), C.float(cfg.RMSEps)) != 0
	if ok {
		ok = C.ds4_metal_matmul_f16_tensor(pg.outputPre.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.tensor("output_hc_fn.weight")), C.uint64_t(hcDim), C.uint64_t(cfg.NHC), pg.flatHC.ptr, 1) != 0
	}
	if ok {
		ok = C.ds4_metal_output_hc_weights_tensor(pg.outputWeights.ptr, pg.outputPre.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.tensor("output_hc_scale.weight")), tensorOffset(me.tensor("output_hc_base.weight")),
			C.uint32_t(cfg.NHC), C.float(cfg.HCEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_hc_weighted_sum_tensor(pg.outputEmbd.ptr, lastHC.ptr, pg.outputWeights.ptr, C.uint32_t(cfg.NEmbd), C.uint32_t(cfg.NHC)) != 0
	}
	if ok {
		ok = C.ds4_metal_rms_norm_weight_tensor(pg.outputNorm.ptr, pg.outputEmbd.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.tensor("output_norm.weight")), C.uint32_t(cfg.NEmbd), C.float(cfg.RMSEps)) != 0
	}
	if ok {
		ok = C.ds4_metal_matmul_q8_0_tensor(pg.logits.ptr, me.mapPtr, C.uint64_t(me.modelSize),
			tensorOffset(me.tensor("output.weight")), C.uint64_t(cfg.NEmbd), C.uint64_t(cfg.NVocab), pg.outputNorm.ptr, 1) != 0
	}
	return ok
}

func (me *MetalEngine) readPrefillCaches(s *Session, nTokens int) bool {
	cfg := s.Engine.Config
	pg := &me.prefill
	rowBytes := uint64(cfg.NHeadDim * 4)
	for il := 0; il < cfg.NLayer; il++ {
		lc := &s.KV.Layer[il]
		nRaw := nTokens
		if nRaw > lc.CapRaw {
			nRaw = lc.CapRaw
		}
		if nRaw > 0 {
			rawOff := uint64(nTokens-nRaw) * rowBytes
			if C.ds4_metal_tensor_read(pg.rawCache[il].ptr, C.uint64_t(rawOff), unsafe.Pointer(&lc.RawKV[0]), C.uint64_t(nRaw*cfg.NHeadDim*4)) == 0 {
				return false
			}
		}
		lc.NRaw = nRaw
		lc.RawWrite = nRaw % lc.CapRaw

		ratio := layerCompressRatio(il)
		if ratio == 0 {
			continue
		}
		nComp := nTokens / ratio
		if nComp > lc.CompCap {
			nComp = lc.CompCap
		}
		if nComp > 0 && C.ds4_metal_tensor_read(pg.attnCompCache[il].ptr, 0, unsafe.Pointer(&lc.CompKV[0]), C.uint64_t(nComp*cfg.NHeadDim*4)) == 0 {
			return false
		}
		lc.NComp = nComp
		lc.CompWrite = nComp % lc.CompCap
		if len(lc.CompStateKV) > 0 && C.ds4_metal_tensor_read(pg.attnStateKV[il].ptr, 0, unsafe.Pointer(&lc.CompStateKV[0]), C.uint64_t(len(lc.CompStateKV)*4)) == 0 {
			return false
		}
		if len(lc.CompStateScore) > 0 && C.ds4_metal_tensor_read(pg.attnStateScore[il].ptr, 0, unsafe.Pointer(&lc.CompStateScore[0]), C.uint64_t(len(lc.CompStateScore)*4)) == 0 {
			return false
		}

		if ratio != 4 {
			continue
		}
		nIndexComp := nTokens / ratio
		if nIndexComp > lc.CompCap {
			nIndexComp = lc.CompCap
		}
		if nIndexComp > 0 && C.ds4_metal_tensor_read(pg.indexCompCache[il].ptr, 0, unsafe.Pointer(&lc.IndexCompKV[0]), C.uint64_t(nIndexComp*cfg.NIndexerHeadDim*4)) == 0 {
			return false
		}
		lc.NIndexComp = nIndexComp
		lc.IndexCompWrite = nIndexComp % lc.CompCap
		if len(lc.IndexStateKV) > 0 && C.ds4_metal_tensor_read(pg.indexStateKV[il].ptr, 0, unsafe.Pointer(&lc.IndexStateKV[0]), C.uint64_t(len(lc.IndexStateKV)*4)) == 0 {
			return false
		}
		if len(lc.IndexStateScore) > 0 && C.ds4_metal_tensor_read(pg.indexStateScore[il].ptr, 0, unsafe.Pointer(&lc.IndexStateScore[0]), C.uint64_t(len(lc.IndexStateScore)*4)) == 0 {
			return false
		}
	}
	return true
}

func (me *MetalEngine) matvecQ8_0(out []float32, tensorName string, x []float32, inDim, outDim int) bool {
	return me.matvecQ8_0At(out, tensorName, 0, x, inDim, outDim)
}

func (me *MetalEngine) matvecQ8_0At(out []float32, tensorName string, byteOffset uint64, x []float32, inDim, outDim int) bool {
	if me == nil || !me.ready || inDim <= 0 || outDim <= 0 || len(x) < inDim || len(out) < outDim {
		return false
	}
	if !me.strict && outDim < me.q8MinRows {
		return false
	}
	t := me.model.Tensors[tensorName]
	if t == nil || t.Type != TensorQ8_0 || inDim%32 != 0 {
		return false
	}
	me.mu.Lock()
	defer me.mu.Unlock()
	if !me.ready {
		return false
	}
	xBytes := uint64(inDim * 4)
	outBytes := uint64(outDim * 4)
	if !me.ensureTensor(&me.x, xBytes) || !me.ensureTensor(&me.out, outBytes) {
		return false
	}
	if C.ds4_metal_tensor_write(me.x.ptr, 0, unsafe.Pointer(&x[0]), C.uint64_t(xBytes)) == 0 {
		return false
	}
	weightOffset := t.AbsOffset + byteOffset
	ok := C.ds4_metal_matmul_q8_0_tensor(
		me.out.ptr,
		me.mapPtr,
		C.uint64_t(me.modelSize),
		C.uint64_t(weightOffset),
		C.uint64_t(inDim),
		C.uint64_t(outDim),
		me.x.ptr,
		1,
	) != 0
	if !ok {
		return false
	}
	if C.ds4_metal_tensor_read(me.out.ptr, 0, unsafe.Pointer(&out[0]), C.uint64_t(outBytes)) == 0 {
		return false
	}
	runtime.KeepAlive(me.model.data)
	runtime.KeepAlive(x)
	runtime.KeepAlive(out)
	return true
}

func (me *MetalEngine) matvecQ8_0Grouped(out []float32, tensorName string, x []float32, inDim, outDim, groupSize int) bool {
	if groupSize <= 0 || inDim%groupSize != 0 {
		return false
	}
	groupDim := inDim / groupSize
	rank := outDim
	nBlocks := (groupDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size
	for g := 0; g < groupSize; g++ {
		x0 := g * groupDim
		o0 := g * rank
		if !me.matvecQ8_0At(out[o0:o0+rank], tensorName, uint64(g*rank*rowBytes), x[x0:x0+groupDim], groupDim, rank) {
			return false
		}
	}
	return true
}

func (me *MetalEngine) expertForward(ds *DecodeState, layer *LayerWeights, experts []expertScore, il int) []bool {
	if me == nil || !me.ready || !me.enableMoE || ds == nil || len(experts) == 0 {
		return nil
	}
	cfg := ds.Cfg()
	ed := detectExpertDims(layer, cfg)
	if ed.outDim == 0 || len(layer.FfnGateExps) == 0 || len(layer.FfnUpExps) == 0 || len(layer.FfnDownExps) == 0 {
		return nil
	}
	prefix := fmt.Sprintf("blk.%d.", il)
	gateT := me.model.Tensors[prefix+"ffn_gate_exps.weight"]
	upT := me.model.Tensors[prefix+"ffn_up_exps.weight"]
	downT := me.model.Tensors[prefix+"ffn_down_exps.weight"]
	if gateT == nil || upT == nil || downT == nil {
		return nil
	}
	if gateT.Type != upT.Type {
		return nil
	}
	switch gateT.Type {
	case TensorIQ2XXS, TensorQ2_K, TensorQ4_K:
	default:
		return nil
	}
	switch downT.Type {
	case TensorQ2_K, TensorQ4_K:
	default:
		return nil
	}

	me.mu.Lock()
	defer me.mu.Unlock()
	if !me.ready {
		return nil
	}

	nExp := len(experts)
	xBytes := uint64(cfg.NEmbd * 4)
	midBytes := uint64(nExp * ed.outDim * 4)
	outBytes := uint64(cfg.NEmbd * 4)
	expertsBytes := uint64(nExp * cfg.NEmbd * 4)
	if !me.ensureTensor(&me.moeX, xBytes) ||
		!me.ensureTensor(&me.moeGate, midBytes) ||
		!me.ensureTensor(&me.moeUp, midBytes) ||
		!me.ensureTensor(&me.moeMid, midBytes) ||
		!me.ensureTensor(&me.moeOut, outBytes) ||
		!me.ensureTensor(&me.moeSelected, uint64(nExp*4)) ||
		!me.ensureTensor(&me.moeWeights, uint64(nExp*4)) {
		return nil
	}
	if nExp > 1 && !me.ensureTensor(&me.moeExps, expertsBytes) {
		return nil
	}

	var selected [NExpertUsed]C.int
	var weights [NExpertUsed]float32
	for i, exp := range experts {
		selected[i] = C.int(exp.idx)
		weights[i] = exp.score
	}
	if C.ds4_metal_tensor_write(me.moeX.ptr, 0, unsafe.Pointer(&ds.FfnNormed[0]), C.uint64_t(xBytes)) == 0 ||
		C.ds4_metal_tensor_write(me.moeSelected.ptr, 0, unsafe.Pointer(&selected[0]), C.uint64_t(nExp*4)) == 0 ||
		C.ds4_metal_tensor_write(me.moeWeights.ptr, 0, unsafe.Pointer(&weights[0]), C.uint64_t(nExp*4)) == 0 {
		return nil
	}

	gateExpertBytes := uint64(len(layer.FfnGateExps) / cfg.NExpert)
	downExpertBytes := uint64(len(layer.FfnDownExps) / cfg.NExpert)
	ok := C.ds4_metal_routed_moe_one_tensor(
		me.moeOut.ptr,
		me.moeGate.ptr,
		me.moeUp.ptr,
		me.moeMid.ptr,
		me.moeExps.ptr,
		me.mapPtr,
		C.uint64_t(me.modelSize),
		C.uint64_t(gateT.AbsOffset),
		C.uint64_t(upT.AbsOffset),
		C.uint64_t(downT.AbsOffset),
		C.uint32_t(gateT.Type),
		C.uint32_t(downT.Type),
		C.uint64_t(gateExpertBytes),
		C.uint64_t(ed.gateRowBytes),
		C.uint64_t(downExpertBytes),
		C.uint64_t(ed.downRowBytes),
		C.uint32_t(cfg.NEmbd),
		C.uint32_t(ed.outDim),
		C.uint32_t(cfg.NEmbd),
		me.moeSelected.ptr,
		me.moeWeights.ptr,
		C.uint32_t(nExp),
		C.float(cfg.SwiGLUClampExp),
		me.moeX.ptr,
	) != 0
	if !ok {
		return nil
	}
	if C.ds4_metal_tensor_read(me.moeOut.ptr, 0, unsafe.Pointer(&ds.RoutedOut[0]), C.uint64_t(outBytes)) == 0 {
		return nil
	}

	handled := make([]bool, nExp)
	for i := range handled {
		handled[i] = true
	}
	runtime.KeepAlive(me.model.data)
	runtime.KeepAlive(ds)
	return handled
}

func metalGPUClose(g interface{}) bool {
	me, ok := g.(*MetalEngine)
	if ok {
		me.Close()
	}
	return ok
}

func metalGPUReady(g interface{}) (bool, bool) {
	me, ok := g.(*MetalEngine)
	return ok && me.ready, ok
}

func metalGPUMatvecQ8_0(g interface{}, out []float32, tensorName string, x []float32, inDim, outDim int) (bool, bool) {
	me, ok := g.(*MetalEngine)
	if !ok {
		return false, false
	}
	return me.matvecQ8_0(out, tensorName, x, inDim, outDim), true
}

func metalGPUMatvecQ8_0Grouped(g interface{}, out []float32, tensorName string, x []float32, inDim, outDim, groupSize int) (bool, bool) {
	me, ok := g.(*MetalEngine)
	if !ok {
		return false, false
	}
	return me.matvecQ8_0Grouped(out, tensorName, x, inDim, outDim, groupSize), true
}

func metalGPUExpertForward(g interface{}, ds *DecodeState, layer *LayerWeights, experts []expertScore, il int) ([]bool, bool) {
	me, ok := g.(*MetalEngine)
	if !ok {
		return nil, false
	}
	return me.expertForward(ds, layer, experts, il), true
}

func metalGPUPrefill(g interface{}, s *Session, tokens []int) (bool, bool) {
	me, ok := g.(*MetalEngine)
	if !ok {
		return false, false
	}
	return me.prefillGraph(s, tokens), true
}

func metalGPUSerializes(g interface{}) bool {
	_, ok := g.(*MetalEngine)
	return ok
}
