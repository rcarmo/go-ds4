package gpu

import (
	"fmt"
	"os"
	"strconv"
	"unsafe"
)

// CompactExpertCache is a budgeted LRU cache of individual routed-expert weight
// slices in VRAM. It mirrors the useful mlx-flash idea for CUDA strict mode:
// selected expert slices stay resident across tokens/layers, per-token batched
// buffers are assembled with D->D copies, and H->D happens only on misses.
type CompactExpertCache struct {
	budgetBytes uint64
	liveBytes   uint64
	peakBytes   uint64
	clock       uint64
	entries     map[compactExpertKey]*compactExpertEntry

	Hits        uint64
	Misses      uint64
	Evictions   uint64
	UploadBytes uint64
	DtoDBytes   uint64
}

type compactExpertKind uint8

const (
	CompactExpertGate compactExpertKind = iota
	CompactExpertUp
	CompactExpertDown
)

type compactExpertKey struct {
	Layer  int
	Expert int
	Kind   compactExpertKind
}

type compactExpertEntry struct {
	ptr      CUdeviceptr
	bytes    uint64
	lastUsed uint64
}

func CompactExpertCacheBudgetFromEnv(defaultMB uint64) uint64 {
	mb := defaultMB
	if s := os.Getenv("DS4_CUDA_COMPACT_EXPERT_CACHE_MB"); s != "" {
		if v, err := strconv.ParseUint(s, 10, 64); err == nil {
			mb = v
		}
	}
	return mb * 1024 * 1024
}

func NewCompactExpertCache(budgetBytes uint64) *CompactExpertCache {
	if budgetBytes == 0 {
		return nil
	}
	return &CompactExpertCache{
		budgetBytes: budgetBytes,
		entries:     make(map[compactExpertKey]*compactExpertEntry, 1024),
	}
}

func (c *CompactExpertCache) BudgetBytes() uint64 {
	if c == nil {
		return 0
	}
	return c.budgetBytes
}
func (c *CompactExpertCache) LiveBytes() uint64 {
	if c == nil {
		return 0
	}
	return c.liveBytes
}
func (c *CompactExpertCache) PeakBytes() uint64 {
	if c == nil {
		return 0
	}
	return c.peakBytes
}
func (c *CompactExpertCache) EntryCount() int {
	if c == nil {
		return 0
	}
	return len(c.entries)
}

func (c *CompactExpertCache) StatsString() string {
	if c == nil {
		return "compact expert cache disabled"
	}
	return fmt.Sprintf("compact expert cache %.1f/%.1f MB live, %.1f MB peak, entries %d, hits %d, misses %d, evictions %d, HtoD %.1f MB, DtoD %.1f MB",
		float64(c.liveBytes)/(1024*1024),
		float64(c.budgetBytes)/(1024*1024),
		float64(c.peakBytes)/(1024*1024),
		len(c.entries), c.Hits, c.Misses, c.Evictions,
		float64(c.UploadBytes)/(1024*1024),
		float64(c.DtoDBytes)/(1024*1024))
}

func (c *CompactExpertCache) evictOne() bool {
	var victimKey compactExpertKey
	var victim *compactExpertEntry
	for k, e := range c.entries {
		if victim == nil || e.lastUsed < victim.lastUsed {
			victimKey, victim = k, e
		}
	}
	if victim == nil {
		return false
	}
	CuMemFreeRaw(victim.ptr)
	if victim.bytes <= c.liveBytes {
		c.liveBytes -= victim.bytes
	} else {
		c.liveBytes = 0
	}
	delete(c.entries, victimKey)
	c.Evictions++
	return true
}

func (c *CompactExpertCache) evictUntil(needed uint64) {
	for c.liveBytes+needed > c.budgetBytes && len(c.entries) != 0 {
		if !c.evictOne() {
			return
		}
	}
}

// CopyTo copies an expert slice into dst. Cache hits use D->D from resident
// VRAM. Cache misses upload directly into the per-token batch buffer so a low
// hit-rate cache does not pay H->D to cache plus D->D to batch.
func (c *CompactExpertCache) CopyTo(dst CUdeviceptr, layer, expert int, kind compactExpertKind, data []byte) error {
	if c == nil {
		if len(data) == 0 {
			return nil
		}
		CuMemcpyHtoDRaw(dst, unsafe.Pointer(&data[0]), uint64(len(data)))
		return nil
	}
	if len(data) == 0 {
		return nil
	}
	key := compactExpertKey{Layer: layer, Expert: expert, Kind: kind}
	entry := c.entries[key]
	if entry != nil {
		c.Hits++
		entry.lastUsed = c.nextClock()
	} else {
		c.Misses++
		bytes := uint64(len(data))
		CuMemcpyHtoDRaw(dst, unsafe.Pointer(&data[0]), bytes)
		c.UploadBytes += bytes
		if bytes > c.budgetBytes {
			return nil
		}
		c.evictUntil(bytes)
		if c.liveBytes+bytes > c.budgetBytes {
			return nil
		}
		var ptr CUdeviceptr
		var err error
		for attempts := 0; attempts < 8; attempts++ {
			err = CuMemAllocRaw(&ptr, bytes)
			if err == nil {
				break
			}
			if !c.evictOne() {
				break
			}
		}
		if err != nil {
			return nil
		}
		if err := CuMemcpyDtoDRaw(ptr, dst, bytes); err != nil {
			CuMemFreeRaw(ptr)
			return nil
		}
		c.DtoDBytes += bytes
		entry = &compactExpertEntry{ptr: ptr, bytes: bytes, lastUsed: c.nextClock()}
		c.entries[key] = entry
		c.liveBytes += bytes
		if c.liveBytes > c.peakBytes {
			c.peakBytes = c.liveBytes
		}
		return nil
	}
	if err := CuMemcpyDtoDRaw(dst, entry.ptr, entry.bytes); err != nil {
		return err
	}
	c.DtoDBytes += entry.bytes
	return nil
}

func (c *CompactExpertCache) nextClock() uint64 {
	c.clock++
	return c.clock
}

func (c *CompactExpertCache) Free() {
	if c == nil {
		return
	}
	for _, e := range c.entries {
		if e.ptr != 0 {
			CuMemFreeRaw(e.ptr)
		}
	}
	c.entries = nil
	c.liveBytes = 0
}
