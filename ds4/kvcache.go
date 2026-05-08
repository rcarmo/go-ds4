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
	IndexCompKV        []float32 // [compCap, NIndexerHeadDim]
	NIndexComp         int
	IndexStateKV       []float32
	IndexStateScore    []float32
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
			lc.CompCap = ctxSize/ratio + 2
			lc.CompKV = make([]float32, lc.CompCap*NHeadDim)
			// Compressor state: ratio entries of KV + score
			lc.CompStateKV = make([]float32, ratio*NHeadDim)
			lc.CompStateScore = make([]float32, ratio)

			if ratio == 4 {
				// Indexer
				lc.IndexCompKV = make([]float32, lc.CompCap*NIndexerHeadDim)
				lc.IndexStateKV = make([]float32, ratio*NIndexerHeadDim)
				lc.IndexStateScore = make([]float32, ratio)
			}
		}
	}
	return kv
}

// layerCompressRatio returns the KV compression ratio for a given layer.
// Mirrors ds4_layer_compress_ratio() in ds4.c.
func layerCompressRatio(il int) int {
	// Layer 0: no compression (dense attention)
	// Layers 1-3: ratio 4 (with indexer)
	// Layers 4+: ratio 2 (compressed, no indexer)
	// This is model-specific — should match the GGUF metadata.
	if il == 0 {
		return 0
	}
	if il <= 3 {
		return 4
	}
	return 2
}

// PushRawKV appends a new KV row to the sliding window (circular buffer).
func (lc *LayerCache) PushRawKV(kv []float32) {
	// Circular: overwrite oldest if full
	idx := lc.NRaw % lc.CapRaw
	copy(lc.RawKV[idx*NHeadDim:(idx+1)*NHeadDim], kv)
	lc.NRaw++
}
