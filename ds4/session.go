package ds4

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/gpu"
)

// Session holds the mutable inference state for one generation timeline.
type Session struct {
	Engine  *Engine
	KV      *KVCache
	Decode  *DecodeState
	Tokens  []int // full token history
	Pos     int   // current position in sequence
	CtxSize int
	Logits  []float32 // [NVocab] last logits
}

// Engine holds the loaded model and weights.
type Engine struct {
	Model       *GGUFModel
	Weights     *Weights
	Vocab       *Vocab
	Budget      *MemoryBudget
	Streamer    *DiskStreamer // non-nil when StreamExperts is enabled
	FastExperts bool          // use top-4 instead of top-6
	GPU         interface{}   // *gpu.GPUEngine when GPU available (avoid import cycle)
}

// EngineOptions configures engine loading.
type EngineOptions struct {
	ModelPath     string
	MaxRSSMB      int  // 0 = unlimited
	PinNonExpert  bool // mlock non-expert weights (~6.5 GB)
	EvictExperts  bool // MADV_DONTNEED cold experts after each layer
	StreamExperts bool // read expert weights from disk instead of mmap
	FastExperts   bool // use top-4 instead of top-6 experts (faster, slight quality loss)
	UseGPU        bool // attempt Vulkan GPU acceleration for dense projections
}

// OpenEngine loads a GGUF model with sensible defaults for low-memory operation.
// Pins non-expert weights (~6.5 GB) and evicts cold experts after each layer,
// keeping RSS at ~7 GB even with a 128 GB model.
func OpenEngine(modelPath string) (*Engine, error) {
	return OpenEngineWithOptions(EngineOptions{
		ModelPath:    modelPath,
		PinNonExpert: true,
		EvictExperts: true,
	})
}

// OpenEngineWithOptions loads a model with memory budget controls.
func OpenEngineWithOptions(opts EngineOptions) (*Engine, error) {
	m, err := OpenGGUF(opts.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}

	w, err := BindWeights(m)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("bind weights: %w", err)
	}

	v, err := LoadVocab(m)
	if err != nil {
		m.Close()
		return nil, fmt.Errorf("load vocab: %w", err)
	}

	budget := &MemoryBudget{
		MaxResidentMB: opts.MaxRSSMB,
		PinNonExpert:  opts.PinNonExpert,
		EvictAfterUse: opts.EvictExperts,
	}
	if err := budget.ApplyBudget(m); err != nil {
		m.Close()
		return nil, fmt.Errorf("apply budget: %w", err)
	}

	e := &Engine{Model: m, Weights: w, Vocab: v, Budget: budget, FastExperts: opts.FastExperts}

	// Open disk streamer if requested
	if opts.StreamExperts {
		streamer, err := NewDiskStreamer(opts.ModelPath, 64)
		if err != nil {
			m.Close()
			return nil, fmt.Errorf("open streamer: %w", err)
		}
		e.Streamer = streamer
	}

	// Initialize GPU if requested
	if opts.UseGPU {
		if err := e.InitGPU(); err != nil {
			fmt.Printf("[gpu] init failed (CPU fallback): %v\n", err)
		}
	}

	return e, nil
}

// Close releases the model memory mapping, streamer, and GPU resources.
func (e *Engine) Close() {
	if e.GPU != nil {
		if g, ok := e.GPU.(*gpu.GPUEngine); ok {
			g.Close()
		}
		e.GPU = nil
	}
	if e.Streamer != nil {
		e.Streamer.Close()
	}
	if e.Model != nil {
		e.Model.Close()
	}
}

// NewSession creates a new inference session with the given context size.
func (e *Engine) NewSession(ctxSize int) *Session {
	ds := NewDecodeState(ctxSize)
	ds.Engine = e
	return &Session{
		Engine:  e,
		KV:      NewKVCache(ctxSize),
		Decode:  ds,
		Tokens:  make([]int, 0, ctxSize),
		CtxSize: ctxSize,
		Logits:  make([]float32, NVocab),
	}
}

// Eval processes one token: runs the full forward pass and produces logits.
func (s *Session) Eval(token int) {
	if token < 0 || token >= NVocab {
		return // skip invalid tokens
	}
	s.Tokens = append(s.Tokens, token)

	// Embed token: look up in token_embd (F16)
	embF16 := s.Engine.Weights.TokenEmbd
	embRowBytes := NEmbd * 2 // F16 = 2 bytes per element
	embOff := token * embRowBytes
	embRow := embF16[embOff : embOff+embRowBytes]

	// Dequantize F16 → F32 into HC stream 0
	embU16 := tensorU16Unsafe(embRow)
	for i := 0; i < NEmbd; i++ {
		s.Decode.CurHC[i] = F16ToF32(embU16[i])
	}
	// Zero other HC streams
	for i := NEmbd; i < hcDim; i++ {
		s.Decode.CurHC[i] = 0
	}

	// Run all 43 layers
	nExperts := NExpertUsed
	if s.Engine.FastExperts {
		nExperts = NExpertUsedFast
	}
	for il := 0; il < NLayer; il++ {
		layerForwardDecode(
			s.Decode,
			&s.Engine.Weights.Layer[il],
			&s.KV.Layer[il],
			s.Engine.Model,
			s.Engine.Budget,
			s.Engine.Streamer,
			s.Pos, il, token, nExperts,
		)
	}

	// Output logits
	outputLogits(s.Decode, s.Logits, s.Decode.CurHC, s.Engine.Weights)

	s.Pos++
}

// Generate runs autoregressive generation for n tokens.
// Calls emit for each generated token. Returns on EOS or n tokens.
func (s *Session) Generate(prompt []int, n int, temperature float32, topK int,
	emit func(token int)) {

	// Prefill: eval all prompt tokens
	for _, t := range prompt {
		s.Eval(t)
	}

	// Decode
	for i := 0; i < n; i++ {
		token := Sample(s.Logits, temperature, topK, 0, 0)
		if token == s.Engine.Vocab.EOS {
			break
		}
		emit(token)
		s.Eval(token)
	}
}

// scored holds a token index and its score for sampling.
type scored struct {
	idx int
	val float32
}

// Sample selects a token from logits with temperature, top-k, top-p, min-p.
func Sample(logits []float32, temperature float32, topK int, topP, minP float32) int {
	if temperature <= 0 {
		return Argmax(logits)
	}

	// Top-K
	n := len(logits)
	if topK <= 0 || topK > n {
		topK = n
	}

	all := make([]scored, n)
	for i, v := range logits {
		all[i] = scored{i, v}
	}

	// Partial sort: find top-K
	if topK < n {
		partialTopK(all, topK)
		all = all[:topK]
	}

	// Temperature scaling + softmax
	maxV := all[0].val
	for _, s := range all[1:] {
		if s.val > maxV {
			maxV = s.val
		}
	}
	sum := float32(0)
	for i := range all {
		e := float32(math.Exp(float64((all[i].val - maxV) / temperature)))
		all[i].val = e
		sum += e
	}
	for i := range all {
		all[i].val /= sum
	}

	// Min-P filtering
	if minP > 0 {
		threshold := all[0].val * minP // relative to max prob
		filtered := all[:0]
		for _, s := range all {
			if s.val >= threshold {
				filtered = append(filtered, s)
			}
		}
		if len(filtered) > 0 {
			all = filtered
		}
	}

	// Top-P (nucleus) filtering
	if topP > 0 && topP < 1 {
		cumul := float32(0)
		cutoff := len(all)
		for i, s := range all {
			cumul += s.val
			if cumul >= topP {
				cutoff = i + 1
				break
			}
		}
		all = all[:cutoff]
	}

	// Re-normalize
	sum = 0
	for _, s := range all {
		sum += s.val
	}

	// Sample
	r := rand.Float32() * sum
	cumul := float32(0)
	for _, s := range all {
		cumul += s.val
		if cumul >= r {
			return s.idx
		}
	}
	return all[len(all)-1].idx
}

// partialTopK partially sorts so that the top-K elements are at the front.
func partialTopK(all []scored, k int) {
	// Simple approach: nth_element-like partial sort
	// For production, use introselect. For now, just sort the top.
	for i := 0; i < k && i < len(all); i++ {
		bestIdx := i
		for j := i + 1; j < len(all); j++ {
			if all[j].val > all[bestIdx].val {
				bestIdx = j
			}
		}
		all[i], all[bestIdx] = all[bestIdx], all[i]
	}
}

// Rewind resets the session to a given position (for prefix reuse).
func (s *Session) Rewind(pos int) {
	if pos < s.Pos {
		s.Pos = pos
		s.Tokens = s.Tokens[:pos]
	}
}

// Invalidate resets the session completely.
func (s *Session) Invalidate() {
	s.Pos = 0
	s.Tokens = s.Tokens[:0]
	s.KV = NewKVCache(s.CtxSize)
}

// SavePayload writes the session KV state to a writer.
func (s *Session) SavePayload(w io.Writer) error {
	// Header
	if err := binary.Write(w, binary.LittleEndian, uint32(0x34565344)); err != nil { // "DSV4"
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(4)); err != nil { // version
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(s.Pos)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(s.CtxSize)); err != nil {
		return err
	}

	// Per-layer KV data
	for il := 0; il < NLayer; il++ {
		lc := &s.KV.Layer[il]
		if err := binary.Write(w, binary.LittleEndian, uint32(lc.NRaw)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint32(lc.RawWrite)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, lc.RawKV); err != nil {
			return err
		}
		if lc.CompressRatio > 0 {
			if err := binary.Write(w, binary.LittleEndian, uint32(lc.NComp)); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, uint32(lc.CompWrite)); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, lc.CompKV); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, lc.CompStateKV); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, lc.CompStateScore); err != nil {
				return err
			}
			if lc.CompressRatio == 4 {
				if err := binary.Write(w, binary.LittleEndian, uint32(lc.NIndexComp)); err != nil {
					return err
				}
				if err := binary.Write(w, binary.LittleEndian, uint32(lc.IndexCompWrite)); err != nil {
					return err
				}
				if err := binary.Write(w, binary.LittleEndian, lc.IndexCompKV); err != nil {
					return err
				}
				if err := binary.Write(w, binary.LittleEndian, lc.IndexStateKV); err != nil {
					return err
				}
				if err := binary.Write(w, binary.LittleEndian, lc.IndexStateScore); err != nil {
					return err
				}
			}
		}
	}

	// Token history
	if err := binary.Write(w, binary.LittleEndian, uint32(len(s.Tokens))); err != nil {
		return err
	}
	for _, t := range s.Tokens {
		if err := binary.Write(w, binary.LittleEndian, int32(t)); err != nil {
			return err
		}
	}

	return nil
}

// LoadPayload restores session KV state from a reader.
func (s *Session) LoadPayload(r io.Reader) error {
	var magic, version, pos, ctxSize uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return err
	}
	if magic != 0x34565344 {
		return fmt.Errorf("bad session magic 0x%08x", magic)
	}
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return err
	}
	if version != 1 && version != 2 && version != 3 && version != 4 {
		return fmt.Errorf("unsupported session version %d", version)
	}
	if err := binary.Read(r, binary.LittleEndian, &pos); err != nil {
		return err
	}
	if err := binary.Read(r, binary.LittleEndian, &ctxSize); err != nil {
		return err
	}

	if s.CtxSize != int(ctxSize) || s.KV == nil {
		s.CtxSize = int(ctxSize)
		s.KV = NewKVCache(s.CtxSize)
	}
	s.Pos = int(pos)

	for il := 0; il < NLayer; il++ {
		lc := &s.KV.Layer[il]
		var nRaw uint32
		if err := binary.Read(r, binary.LittleEndian, &nRaw); err != nil {
			return err
		}
		lc.NRaw = int(nRaw)
		if version >= 3 {
			var rawWrite uint32
			if err := binary.Read(r, binary.LittleEndian, &rawWrite); err != nil {
				return err
			}
			lc.RawWrite = int(rawWrite) % lc.CapRaw
		} else {
			lc.RawWrite = lc.NRaw % lc.CapRaw
		}
		if err := binary.Read(r, binary.LittleEndian, lc.RawKV); err != nil {
			return err
		}
		if lc.CompressRatio > 0 {
			var nComp uint32
			if err := binary.Read(r, binary.LittleEndian, &nComp); err != nil {
				return err
			}
			lc.NComp = int(nComp)
			if version >= 4 {
				var compWrite uint32
				if err := binary.Read(r, binary.LittleEndian, &compWrite); err != nil {
					return err
				}
				if lc.CompCap > 0 {
					lc.CompWrite = int(compWrite) % lc.CompCap
				}
			} else {
				if lc.CompCap > 0 {
					lc.CompWrite = lc.NComp % lc.CompCap
				}
			}
			if err := binary.Read(r, binary.LittleEndian, lc.CompKV); err != nil {
				return err
			}
			if err := binary.Read(r, binary.LittleEndian, lc.CompStateKV); err != nil {
				return err
			}
			if err := binary.Read(r, binary.LittleEndian, lc.CompStateScore); err != nil {
				return err
			}
			if lc.CompressRatio == 4 {
				if version >= 2 {
					var nIndexComp uint32
					if err := binary.Read(r, binary.LittleEndian, &nIndexComp); err != nil {
						return err
					}
					lc.NIndexComp = int(nIndexComp)
					if version >= 4 {
						var indexWrite uint32
						if err := binary.Read(r, binary.LittleEndian, &indexWrite); err != nil {
							return err
						}
						if lc.CompCap > 0 {
							lc.IndexCompWrite = int(indexWrite) % lc.CompCap
						}
					} else {
						if lc.CompCap > 0 {
							lc.IndexCompWrite = lc.NIndexComp % lc.CompCap
						}
					}
					if err := binary.Read(r, binary.LittleEndian, lc.IndexCompKV); err != nil {
						return err
					}
					if err := binary.Read(r, binary.LittleEndian, lc.IndexStateKV); err != nil {
						return err
					}
					if err := binary.Read(r, binary.LittleEndian, lc.IndexStateScore); err != nil {
						return err
					}
				} else {
					// v1 payloads had no indexer state
					lc.NIndexComp = 0
				}
			}
		}
	}

	var nTokens uint32
	if err := binary.Read(r, binary.LittleEndian, &nTokens); err != nil {
		return err
	}
	s.Tokens = make([]int, nTokens)
	for i := range s.Tokens {
		var t int32
		if err := binary.Read(r, binary.LittleEndian, &t); err != nil {
			return err
		}
		s.Tokens[i] = int(t)
	}

	return nil
}

func tensorU16Unsafe(data []byte) []uint16 {
	return tensorU16(data)
}

func tensorU16(data []byte) []uint16 {
	if len(data) < 2 {
		return nil
	}
	return (*[1 << 30]uint16)(unsafePtrBytes(data))[:len(data)/2]
}

func unsafePtrBytes(b []byte) unsafe.Pointer {
	return unsafe.Pointer(&b[0])
}
