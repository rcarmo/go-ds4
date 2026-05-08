package ds4

// LayerCache holds the per-layer KV cache state.
type LayerCache struct {
	// Raw SWA (sliding window attention) KV rows, logical ring buffer.
	RawKV    []float32 // [capRaw, NHeadDim]
	NRaw     int       // live rows in raw window (<= CapRaw)
	CapRaw   int       // = NSWA (128) or ctx-dependent
	RawWrite int       // next physical write slot in RawKV

	// Compressed KV (for layers with compress_ratio > 0)
	CompressRatio int       // 0=none, 4=4:1, 128=128:1
	CompKV        []float32 // [compCap, NHeadDim]
	NComp         int       // live compressed rows (<= CompCap)
	CompCap       int
	CompWrite     int // next physical write slot in CompKV

	// Compressor state (learned aggregator)
	CompStateKV    []float32
	CompStateScore []float32

	// Indexer compressed KV (ratio-4 layers only)
	IndexCompKV     []float32 // [compCap, NIndexerHeadDim]
	NIndexComp      int       // live rows (<= CompCap)
	IndexCompWrite  int       // next physical write slot in IndexCompKV
	IndexStateKV    []float32
	IndexStateScore []float32
}

// KVCache holds the full KV cache across all layers.
type KVCache struct {
	Layer [NLayer]LayerCache
}

// NewKVCache allocates a KV cache for the given context size.
func NewKVCache(ctxSize int) *KVCache {
	kv := &KVCache{}
	for il := 0; il < NLayer; il++ {
		lc := &kv.Layer[il]
		lc.CapRaw = NSWA
		lc.RawKV = make([]float32, lc.CapRaw*NHeadDim)

		ratio := layerCompressRatio(il)
		lc.CompressRatio = ratio
		if ratio > 0 {
			coff := 1
			if ratio == 4 {
				coff = 2
			}
			width := coff * NHeadDim
			rows := coff * ratio

			lc.CompCap = ctxSize/ratio + 2
			lc.CompKV = make([]float32, lc.CompCap*NHeadDim)
			lc.CompStateKV = make([]float32, rows*width)
			lc.CompStateScore = make([]float32, rows*width)
			for i := range lc.CompStateScore {
				lc.CompStateScore[i] = -1e30
			}

			if ratio == 4 {
				idxWidth := coff * NIndexerHeadDim
				idxRows := coff * ratio
				lc.IndexCompKV = make([]float32, lc.CompCap*NIndexerHeadDim)
				lc.IndexStateKV = make([]float32, idxRows*idxWidth)
				lc.IndexStateScore = make([]float32, idxRows*idxWidth)
				for i := range lc.IndexStateScore {
					lc.IndexStateScore[i] = -1e30
				}
			}
		}
	}
	return kv
}

// layerCompressRatio returns the KV compression ratio for a given layer.
// Mirrors ds4_layer_compress_ratio() in ds4.c.
func layerCompressRatio(il int) int {
	// DeepSeek V4 Flash layout (from ds4.c):
	// - Layers 0-1: dense attention (no compression)
	// - Even layers >= 2: ratio-4 compression (with indexer)
	// - Odd layers >= 3: ratio-128 compression (no indexer)
	if il < 2 {
		return 0
	}
	if il&1 == 0 {
		return 4
	}
	return 128
}

// PushRawKV appends a new KV row to the raw SWA window using a logical ring.
// Oldest rows are overwritten when full, with no memmove.
func (lc *LayerCache) PushRawKV(kv []float32) {
	idx := lc.RawWrite
	copy(lc.RawKV[idx*NHeadDim:(idx+1)*NHeadDim], kv)
	lc.RawWrite++
	if lc.RawWrite >= lc.CapRaw {
		lc.RawWrite = 0
	}
	if lc.NRaw < lc.CapRaw {
		lc.NRaw++
	}
}

// rawStart returns the physical slot of the oldest row in RawKV.
func (lc *LayerCache) rawStart() int {
	start := lc.RawWrite - lc.NRaw
	if start < 0 {
		start += lc.CapRaw
	}
	return start
}

// RawRow returns row i in chronological order (0=oldest, NRaw-1=newest).
func (lc *LayerCache) RawRow(i int) []float32 {
	idx := lc.rawStart() + i
	if idx >= lc.CapRaw {
		idx -= lc.CapRaw
	}
	return lc.RawKV[idx*NHeadDim : (idx+1)*NHeadDim]
}

// PushCompKV appends one compressed attention KV row using a logical ring.
func (lc *LayerCache) PushCompKV(kv []float32) {
	if lc.CompCap == 0 {
		return
	}
	idx := lc.CompWrite
	copy(lc.CompKV[idx*NHeadDim:(idx+1)*NHeadDim], kv)
	lc.CompWrite++
	if lc.CompWrite >= lc.CompCap {
		lc.CompWrite = 0
	}
	if lc.NComp < lc.CompCap {
		lc.NComp++
	}
}

func (lc *LayerCache) compStart() int {
	start := lc.CompWrite - lc.NComp
	if start < 0 {
		start += lc.CompCap
	}
	return start
}

// CompRow returns compressed attention row i in chronological order.
func (lc *LayerCache) CompRow(i int) []float32 {
	idx := lc.compStart() + i
	if idx >= lc.CompCap {
		idx -= lc.CompCap
	}
	return lc.CompKV[idx*NHeadDim : (idx+1)*NHeadDim]
}

// PushIndexCompKV appends one compressed indexer KV row using a logical ring.
func (lc *LayerCache) PushIndexCompKV(kv []float32) {
	if lc.CompCap == 0 {
		return
	}
	idx := lc.IndexCompWrite
	copy(lc.IndexCompKV[idx*NIndexerHeadDim:(idx+1)*NIndexerHeadDim], kv)
	lc.IndexCompWrite++
	if lc.IndexCompWrite >= lc.CompCap {
		lc.IndexCompWrite = 0
	}
	if lc.NIndexComp < lc.CompCap {
		lc.NIndexComp++
	}
}

func (lc *LayerCache) indexCompStart() int {
	start := lc.IndexCompWrite - lc.NIndexComp
	if start < 0 {
		start += lc.CompCap
	}
	return start
}

// IndexCompRow returns compressed indexer row i in chronological order.
func (lc *LayerCache) IndexCompRow(i int) []float32 {
	idx := lc.indexCompStart() + i
	if idx >= lc.CompCap {
		idx -= lc.CompCap
	}
	return lc.IndexCompKV[idx*NIndexerHeadDim : (idx+1)*NIndexerHeadDim]
}
