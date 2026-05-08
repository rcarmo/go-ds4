package ds4

// LayerCache holds the per-layer KV cache state.
type LayerCache struct {
	// Raw SWA (sliding window attention) KV rows
	RawKV  []float32 // [capRaw, NHeadDim]
	NRaw   int       // live rows in raw window
	CapRaw int       // = NSWA (128) or ctx-dependent

	// Compressed KV (for layers with compress_ratio > 0)
	CompressRatio int       // 0=none, 2=2:1, 4=4:1
	CompKV        []float32 // [compCap, NHeadDim]
	NComp         int       // live compressed rows
	CompCap       int

	// Compressor state (learned aggregator)
	CompStateKV    []float32
	CompStateScore []float32

	// Indexer compressed KV (ratio-4 layers only)
	IndexCompKV     []float32 // [compCap, NIndexerHeadDim]
	NIndexComp      int
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

// PushRawKV appends a new KV row to the sliding window in chronological order.
// Mirrors ds4.c behavior: append until full, then shift left by one row.
func (lc *LayerCache) PushRawKV(kv []float32) {
	if lc.NRaw < lc.CapRaw {
		copy(lc.RawKV[lc.NRaw*NHeadDim:(lc.NRaw+1)*NHeadDim], kv)
		lc.NRaw++
		return
	}
	copy(lc.RawKV[0:(lc.CapRaw-1)*NHeadDim], lc.RawKV[NHeadDim:lc.CapRaw*NHeadDim])
	copy(lc.RawKV[(lc.CapRaw-1)*NHeadDim:lc.CapRaw*NHeadDim], kv)
}

// PushCompKV appends one compressed attention KV row.
func (lc *LayerCache) PushCompKV(kv []float32) {
	if lc.NComp >= lc.CompCap {
		return
	}
	copy(lc.CompKV[lc.NComp*NHeadDim:(lc.NComp+1)*NHeadDim], kv)
	lc.NComp++
}

// PushIndexCompKV appends one compressed indexer KV row.
func (lc *LayerCache) PushIndexCompKV(kv []float32) {
	if lc.NIndexComp >= lc.CompCap {
		return
	}
	copy(lc.IndexCompKV[lc.NIndexComp*NIndexerHeadDim:(lc.NIndexComp+1)*NIndexerHeadDim], kv)
	lc.NIndexComp++
}
