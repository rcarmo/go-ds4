package ds4

import (
	"fmt"
	"io"
	"os"
	"sync"
)

// DiskStreamer provides on-demand disk reads for expert weights,
// avoiding the need to mmap the full 128 GB model into memory.
//
// Non-expert weights (~6.5 GB) remain mmap'd for instant access.
// Expert weights (~120 GB) are read from disk into a reusable buffer
// pool when needed, then released after each layer.
//
// With NVMe SSD (~3-7 GB/s sequential), reading 6 active experts
// per layer costs ~1ms/layer × 43 layers ≈ 50ms/token overhead.
// This enables inference on machines with 8-16 GB RAM.
type DiskStreamer struct {
	file     *os.File
	mu       sync.Mutex
	bufPool  sync.Pool // reusable []byte buffers
	maxBufMB int       // max buffer size in MB

	// Cache: keep the last layer's expert data warm
	cachedLayer int
	cachedData  map[string][]byte // tensor_name → data
}

// NewDiskStreamer opens the model file for streaming reads.
// maxBufMB controls the buffer pool size (default 64 MB).
func NewDiskStreamer(path string, maxBufMB int) (*DiskStreamer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("streamer: open: %w", err)
	}
	if maxBufMB <= 0 {
		maxBufMB = 64
	}
	return &DiskStreamer{
		file:        f,
		maxBufMB:    maxBufMB,
		cachedLayer: -1,
		cachedData:  make(map[string][]byte),
		bufPool: sync.Pool{
			New: func() interface{} {
				// Pre-allocate a buffer large enough for one expert's
				// largest tensor (Q2_K down: ~328 KB per expert)
				buf := make([]byte, 512*1024)
				return &buf
			},
		},
	}, nil
}

// Close releases the file handle.
func (ds *DiskStreamer) Close() {
	if ds.file != nil {
		ds.file.Close()
	}
}

// ReadExpertTensor reads a specific expert's slice from a multi-expert tensor.
// The data is read from disk at the tensor's file offset + expert stride.
// Returns a byte slice (may be from the buffer pool — caller must not retain).
func (ds *DiskStreamer) ReadExpertTensor(t *GGUFTensor, expertIdx int) ([]byte, error) {
	totalBytes := t.DataBytes()
	if totalBytes == 0 {
		return nil, fmt.Errorf("streamer: tensor %s has zero bytes", t.Name)
	}

	bytesPerExpert := totalBytes / uint64(NExpert)
	offset := int64(t.AbsOffset) + int64(expertIdx)*int64(bytesPerExpert)
	size := int(bytesPerExpert)

	// Get a buffer
	buf := ds.getBuffer(size)

	ds.mu.Lock()
	defer ds.mu.Unlock()

	_, err := ds.file.Seek(offset, io.SeekStart)
	if err != nil {
		return nil, fmt.Errorf("streamer: seek %s expert %d: %w", t.Name, expertIdx, err)
	}

	_, err = io.ReadFull(ds.file, buf[:size])
	if err != nil {
		return nil, fmt.Errorf("streamer: read %s expert %d (%d bytes): %w",
			t.Name, expertIdx, size, err)
	}

	return buf[:size], nil
}

// ReadTensorRegion reads an arbitrary region from the model file.
func (ds *DiskStreamer) ReadTensorRegion(offset int64, size int) ([]byte, error) {
	buf := ds.getBuffer(size)

	ds.mu.Lock()
	defer ds.mu.Unlock()

	_, err := ds.file.Seek(offset, io.SeekStart)
	if err != nil {
		return nil, err
	}
	_, err = io.ReadFull(ds.file, buf[:size])
	if err != nil {
		return nil, err
	}
	return buf[:size], nil
}

func (ds *DiskStreamer) getBuffer(size int) []byte {
	bp := ds.bufPool.Get().(*[]byte)
	buf := *bp
	if len(buf) < size {
		buf = make([]byte, size)
	}
	return buf[:size]
}

// ReturnBuffer returns a buffer to the pool for reuse.
func (ds *DiskStreamer) ReturnBuffer(buf []byte) {
	if cap(buf) > 0 {
		b := buf[:cap(buf)]
		ds.bufPool.Put(&b)
	}
}

// StreamingWeights wraps a GGUFModel + DiskStreamer for hybrid access:
// non-expert weights via mmap, expert weights via disk streaming.
type StreamingWeights struct {
	Weights  *Weights // mmap-backed (non-expert tensors valid, expert tensors nil)
	Streamer *DiskStreamer
	Model    *GGUFModel

	// Expert buffer cache: per-layer, reused across tokens
	expertBufs [NLayer]layerExpertBufs
}

type layerExpertBufs struct {
	gate [NExpertUsed][]byte
	up   [NExpertUsed][]byte
	down [NExpertUsed][]byte
}

// LoadStreamingExpert reads one expert's gate/up/down weights from disk
// into the per-layer buffer cache. Returns byte slices for each matrix.
func (sw *StreamingWeights) LoadStreamingExpert(il, slot, expertIdx int) (gate, up, down []byte, err error) {
	prefix := fmt.Sprintf("blk.%d.", il)

	gateTensor := sw.Model.Tensors[prefix+"ffn_gate_exps.weight"]
	upTensor := sw.Model.Tensors[prefix+"ffn_up_exps.weight"]
	downTensor := sw.Model.Tensors[prefix+"ffn_down_exps.weight"]

	if gateTensor == nil || upTensor == nil || downTensor == nil {
		return nil, nil, nil, fmt.Errorf("missing expert tensors for layer %d", il)
	}

	gate, err = sw.Streamer.ReadExpertTensor(gateTensor, expertIdx)
	if err != nil {
		return nil, nil, nil, err
	}
	sw.expertBufs[il].gate[slot] = gate

	up, err = sw.Streamer.ReadExpertTensor(upTensor, expertIdx)
	if err != nil {
		return nil, nil, nil, err
	}
	sw.expertBufs[il].up[slot] = up

	down, err = sw.Streamer.ReadExpertTensor(downTensor, expertIdx)
	if err != nil {
		return nil, nil, nil, err
	}
	sw.expertBufs[il].down[slot] = down

	return gate, up, down, nil
}

// StreamingEstimate returns the estimated memory usage for streaming mode.
func StreamingEstimate(ctxSize int) (mmapMB, bufferMB, totalMB float64) {
	s := EstimateMemory(ctxSize)
	mmapMB = s.NonExpertMB // non-expert weights stay mmap'd
	// Buffer: 6 experts × 3 matrices × max size per expert
	bufferPerExpert := float64(NFFExp*NEmbd*BlockIQ2XXSSize)/float64(QK_K)*2 + // gate+up IQ2
		float64(NEmbd*NFFExp*BlockQ2KSize)/float64(QK_K) // down Q2_K
	bufferMB = float64(NExpertUsed) * bufferPerExpert / (1024 * 1024)
	totalMB = mmapMB + bufferMB + s.KVCacheMB
	return
}
