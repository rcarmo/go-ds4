package ds4

import (
	"fmt"
	"runtime"
	"syscall"
)

// MmapHints configures memory-mapped access patterns for the model.
type MmapHints struct {
	// Sequential prefetch for dense tensors (embeddings, norms, routing)
	// Random access for expert weights (only 6 of 256 touched per token)
	expertTensors map[string]bool
}

// ApplyMmapHints sets madvise hints on the model's memory mapping.
// Dense tensors get MADV_SEQUENTIAL; expert tensors get MADV_RANDOM.
func (m *GGUFModel) ApplyMmapHints() {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		return
	}

	pageSize := syscall.Getpagesize()

	for name, t := range m.Tensors {
		start := t.AbsOffset
		size := t.DataBytes()
		if size == 0 {
			continue
		}

		// Align to page boundaries
		alignedStart := start & ^uint64(pageSize-1)
		alignedEnd := (start + size + uint64(pageSize) - 1) & ^uint64(pageSize-1)
		alignedSize := alignedEnd - alignedStart

		if alignedStart >= uint64(len(m.data)) || alignedSize == 0 {
			continue
		}

		region := m.data[alignedStart : alignedStart+alignedSize]

		// Expert weights get MADV_RANDOM (demand-paged, only 6/256 active)
		isExpert := isExpertTensor(name)
		if isExpert {
			madvise(region, syscall.MADV_RANDOM)
		} else {
			madvise(region, syscall.MADV_SEQUENTIAL)
		}
	}
}

// PrefetchLayer tells the OS to prefetch a layer's non-expert weights.
// Called before processing each layer to warm the cache.
func (m *GGUFModel) PrefetchLayer(il int) {
	prefix := fmt.Sprintf("blk.%d.", il)
	for name, t := range m.Tensors {
		if len(name) < len(prefix) || name[:len(prefix)] != prefix {
			continue
		}
		if isExpertTensor(name) {
			continue // skip experts — they're demand-paged
		}
		start := t.AbsOffset
		size := t.DataBytes()
		if size > 0 && start+size <= uint64(len(m.data)) {
			madvise(m.data[start:start+size], syscall.MADV_WILLNEED)
		}
	}
}

// PrefetchExperts tells the OS to prefetch specific expert weights.
// Called after routing to warm only the selected experts' pages.
func (m *GGUFModel) PrefetchExperts(il int, expertIDs []int) {
	prefix := fmt.Sprintf("blk.%d.", il)
	for name, t := range m.Tensors {
		if len(name) < len(prefix) || name[:len(prefix)] != prefix {
			continue
		}
		if !isExpertTensor(name) {
			continue
		}
		// For expert tensors, compute the slice for each selected expert
		// Expert data is [NFFExp, NExpert*dim] — each expert occupies
		// a contiguous stride of dim elements per output row.
		for _, eid := range expertIDs {
			expertSlice := expertRegion(t, eid)
			if expertSlice.start+expertSlice.size <= uint64(len(m.data)) {
				madvise(m.data[expertSlice.start:expertSlice.start+expertSlice.size],
					syscall.MADV_WILLNEED)
			}
		}
	}
}

type region struct {
	start, size uint64
}

// expertRegion computes the byte range for one expert within a multi-expert tensor.
func expertRegion(t *GGUFTensor, expertIdx int) region {
	// Expert tensors have shape [outDim, NExpert*inDim] or [inDim, NExpert*outDim]
	// Each expert's data is interleaved by output row.
	// For IQ2_XXS: each expert has (inDim/QK_K) blocks per output row, NFFExp rows.
	// Total per expert ≈ NFFExp * (inDim/QK_K) * blockSize bytes.
	ne := t.NumElements()
	totalBytes := t.DataBytes()
	if ne == 0 || totalBytes == 0 {
		return region{}
	}

	bytesPerExpert := totalBytes / uint64(NExpert)
	start := t.AbsOffset + uint64(expertIdx)*bytesPerExpert
	return region{start: start, size: bytesPerExpert}
}

// isExpertTensor returns true for routed expert weight tensors.
func isExpertTensor(name string) bool {
	// Expert tensors contain "exps" in the name
	for i := 0; i+3 < len(name); i++ {
		if name[i:i+4] == "exps" {
			return true
		}
	}
	return false
}

// madvise wraps the syscall, ignoring errors silently.
func madvise(b []byte, advice int) {
	if len(b) == 0 {
		return
	}
	// syscall.Madvise is not available on all platforms
	// Use the raw syscall
	_ = madviseRaw(b, advice)
}
