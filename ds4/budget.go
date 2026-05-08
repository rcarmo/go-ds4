package ds4

import (
	"fmt"
	"runtime"
	"syscall"
)

// MemoryBudget controls how much physical RAM the model keeps resident.
// The model file is always fully mmap'd (virtual address space is cheap),
// but physical page residency is managed via mlock/madvise.
type MemoryBudget struct {
	// MaxResidentMB is the target RSS cap in megabytes.
	// 0 means unlimited (OS manages everything).
	MaxResidentMB int

	// PinNonExpert locks the non-expert weights (projections, norms,
	// shared expert, embeddings, routing) into RAM. These are always
	// accessed every token and amount to ~6.5 GB for DS4 Q2.
	PinNonExpert bool

	// EvictAfterUse calls MADV_DONTNEED on expert pages after each
	// layer's MoE forward pass, returning those pages to the OS.
	// This keeps RSS near the non-expert baseline between layers.
	EvictAfterUse bool

	model *GGUFModel
}

// MemorySummary returns the estimated size breakdown.
type MemorySummary struct {
	NonExpertMB float64 // always-accessed weights
	ExpertMB    float64 // all 256×43 expert matrices
	ActiveSetMB float64 // non-expert + 6/256 experts × 43 layers
	KVCacheMB   float64 // KV cache at given context size
}

// EstimateMemory returns the model's memory breakdown.
func EstimateMemory(ctxSize int) MemorySummary {
	// These are computed from the model constants
	bq8_0 := 36   // Q8_0 block: 36 bytes per 32 elements
	biq2 := 66    // IQ2_XXS: 66 bytes per 256 elements
	bq2k := 84    // Q2_K: 84 bytes per 256 elements

	perLayerNonExpert := float64(0)
	// Attention projections (Q8_0)
	perLayerNonExpert += float64(NEmbd*NLoraQ*bq8_0) / 32           // attn_q_a
	perLayerNonExpert += float64(NLoraQ*NHead*NHeadDim*bq8_0) / 32  // attn_q_b
	perLayerNonExpert += float64(NEmbd*NHeadDim*bq8_0) / 32         // attn_kv
	perLayerNonExpert += float64(NHead*NHeadDim*NLoraO*bq8_0) / 32  // attn_out_a
	perLayerNonExpert += float64(NLoraO*NEmbd*bq8_0) / 32           // attn_out_b
	// Shared expert (Q8_0)
	perLayerNonExpert += float64(NFFExp*NEmbd*bq8_0) / 32 * 3       // gate+up+down
	// HC projections (F16) + norms + routing
	perLayerNonExpert += float64(NHC*NEmbd*(2*NHC+NHC*NHC)) * 2 * 2 // 2 HC blocks × F16
	perLayerNonExpert += float64(NEmbd*NExpert) * 2                  // routing (F16)

	perLayerExpert := float64(0)
	perLayerExpert += float64(NFFExp*NExpert*NEmbd*biq2) / QK_K * 2  // gate+up
	perLayerExpert += float64(NEmbd*NExpert*NFFExp*bq2k) / QK_K      // down

	totalNonExpert := float64(NLayer)*perLayerNonExpert +
		float64(NVocab*NEmbd)*2 +                    // token_embd (F16)
		float64(NVocab*NEmbd*bq8_0)/32               // output (Q8_0)

	totalExpert := float64(NLayer) * perLayerExpert
	activeSet := totalNonExpert + float64(NLayer)*float64(NExpertUsed)/float64(NExpert)*perLayerExpert

	// KV cache: per layer, raw SWA + compressed
	kvPerLayer := float64(NSWA*NHeadDim) * 4 // raw SWA (float32)
	for il := 0; il < NLayer; il++ {
		r := layerCompressRatio(il)
		if r > 0 {
			compCap := ctxSize/r + 2
			kvPerLayer += float64(compCap*NHeadDim) * 4
		}
	}

	MB := 1024.0 * 1024.0
	return MemorySummary{
		NonExpertMB: totalNonExpert / MB,
		ExpertMB:    totalExpert / MB,
		ActiveSetMB: activeSet / MB,
		KVCacheMB:   kvPerLayer / MB,
	}
}

// ApplyBudget configures mlock/madvise for the given memory budget.
func (b *MemoryBudget) ApplyBudget(m *GGUFModel) error {
	b.model = m

	if runtime.GOOS != "linux" {
		return nil // mlock/madvise only on Linux for now
	}

	if b.PinNonExpert {
		if err := b.mlockNonExpert(); err != nil {
			return fmt.Errorf("mlock non-expert: %w", err)
		}
	}

	// Set MADV_RANDOM on all expert tensors (prevents prefetch of cold experts)
	m.ApplyMmapHints()

	return nil
}

// mlockNonExpert pins non-expert tensor pages into RAM.
func (b *MemoryBudget) mlockNonExpert() error {
	pageSize := syscall.Getpagesize()
	locked := 0

	for name, t := range b.model.Tensors {
		if isExpertTensor(name) {
			continue
		}
		start := t.AbsOffset
		size := t.DataBytes()
		if size == 0 {
			continue
		}

		// Align to pages
		alignedStart := start & ^uint64(pageSize-1)
		alignedEnd := (start + size + uint64(pageSize) - 1) & ^uint64(pageSize-1)
		alignedSize := int(alignedEnd - alignedStart)

		if alignedStart >= uint64(len(b.model.data)) {
			continue
		}

		region := b.model.data[alignedStart : alignedStart+uint64(alignedSize)]
		err := syscall.Mlock(region)
		if err != nil {
			return fmt.Errorf("mlock %s (%d bytes): %w", name, alignedSize, err)
		}
		locked += alignedSize
	}

	fmt.Printf("ds4: mlock'd %d MB of non-expert weights\n", locked/1024/1024)
	return nil
}

// EvictLayerExperts releases expert pages for a layer back to the OS.
// Call this after processing each layer's MoE forward pass.
func (b *MemoryBudget) EvictLayerExperts(il int) {
	if !b.EvictAfterUse || b.model == nil {
		return
	}

	prefix := fmt.Sprintf("blk.%d.", il)
	for name, t := range b.model.Tensors {
		if len(name) < len(prefix) || name[:len(prefix)] != prefix {
			continue
		}
		if !isExpertTensor(name) {
			continue
		}
		start := t.AbsOffset
		size := t.DataBytes()
		if size > 0 && start+size <= uint64(len(b.model.data)) {
			// MADV_DONTNEED: release pages, they'll be demand-paged from disk if needed again
			madvise(b.model.data[start:start+size], syscall.MADV_DONTNEED)
		}
	}
}

// EvictColdExperts releases specific expert pages that were NOT selected.
// More surgical than EvictLayerExperts — keeps the selected experts warm
// in case the same experts are picked again next token.
func (b *MemoryBudget) EvictColdExperts(il int, activeExperts []int) {
	if !b.EvictAfterUse || b.model == nil {
		return
	}

	active := make(map[int]bool, len(activeExperts))
	for _, e := range activeExperts {
		active[e] = true
	}

	prefix := fmt.Sprintf("blk.%d.", il)
	for name, t := range b.model.Tensors {
		if len(name) < len(prefix) || name[:len(prefix)] != prefix {
			continue
		}
		if !isExpertTensor(name) {
			continue
		}
		// Evict only the cold experts' regions
		for eid := 0; eid < NExpert; eid++ {
			if active[eid] {
				continue // keep warm
			}
			r := expertRegion(t, eid)
			if r.size > 0 && r.start+r.size <= uint64(len(b.model.data)) {
				madvise(b.model.data[r.start:r.start+r.size], syscall.MADV_DONTNEED)
			}
		}
	}
}
