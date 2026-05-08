package ds4

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"syscall"
	"unsafe"
)

// GGUF v3 magic and version.
const (
	ggufMagic   = 0x46554747 // "GGUF" little-endian
	ggufVersion = 3
)

// GGUF metadata value types.
const (
	ggufUint8   = 0
	ggufInt8    = 1
	ggufUint16  = 2
	ggufInt16   = 3
	ggufUint32  = 4
	ggufInt32   = 5
	ggufFloat32 = 6
	ggufBool    = 7
	ggufString  = 8
	ggufArray   = 9
	ggufUint64  = 10
	ggufInt64   = 11
	ggufFloat64 = 12
)

// GGUFTensor describes one tensor in the GGUF file.
type GGUFTensor struct {
	Name      string
	Type      uint32
	NDim      uint32
	Dims      [8]uint64
	AbsOffset uint64 // byte offset from start of mmap
}

// NumElements returns the total element count.
func (t *GGUFTensor) NumElements() uint64 {
	n := uint64(1)
	for i := uint32(0); i < t.NDim; i++ {
		n *= t.Dims[i]
	}
	return n
}

// DataBytes returns the byte size of this tensor's data.
func (t *GGUFTensor) DataBytes() uint64 {
	bpb, epb := TensorTypeSize(t.Type)
	if epb == 0 {
		return 0
	}
	ne := t.NumElements()
	nBlocks := (ne + uint64(epb) - 1) / uint64(epb)
	return nBlocks * uint64(bpb)
}

// GGUFModel holds a memory-mapped GGUF file and its parsed directory.
type GGUFModel struct {
	data    []byte              // full mmap
	Tensors map[string]*GGUFTensor // name → tensor descriptor
	Meta    map[string]interface{} // metadata key-values (selected)
}

// OpenGGUF opens and parses a GGUF v3 file via mmap.
func OpenGGUF(path string) (*GGUFModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("gguf: stat: %w", err)
	}
	size := fi.Size()
	if size < 32 {
		return nil, errors.New("gguf: file too small")
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("gguf: mmap: %w", err)
	}

	m := &GGUFModel{data: data}
	if err := m.parse(); err != nil {
		syscall.Munmap(data)
		return nil, err
	}
	return m, nil
}

// Close unmaps the model file.
func (m *GGUFModel) Close() {
	if m.data != nil {
		syscall.Munmap(m.data)
		m.data = nil
	}
}

// TensorData returns a byte slice pointing directly into the mmap for a tensor.
func (m *GGUFModel) TensorData(name string) ([]byte, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	end := t.AbsOffset + t.DataBytes()
	if end > uint64(len(m.data)) {
		return nil, fmt.Errorf("gguf: tensor %q extends past file (off=%d size=%d file=%d)",
			name, t.AbsOffset, t.DataBytes(), len(m.data))
	}
	return m.data[t.AbsOffset:end], nil
}

// TensorF32 returns a float32 slice pointing into the mmap (zero-copy).
func (m *GGUFModel) TensorF32(name string) ([]float32, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if t.Type != TensorF32 {
		return nil, fmt.Errorf("gguf: tensor %q type %d, want F32", name, t.Type)
	}
	ne := t.NumElements()
	end := t.AbsOffset + ne*4
	if end > uint64(len(m.data)) {
		return nil, fmt.Errorf("gguf: tensor %q OOB", name)
	}
	ptr := unsafe.Pointer(&m.data[t.AbsOffset])
	return unsafe.Slice((*float32)(ptr), ne), nil
}

// TensorF16 returns a uint16 slice (F16 values) pointing into the mmap.
func (m *GGUFModel) TensorF16(name string) ([]uint16, error) {
	t, ok := m.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	if t.Type != TensorF16 {
		return nil, fmt.Errorf("gguf: tensor %q type %d, want F16", name, t.Type)
	}
	ne := t.NumElements()
	end := t.AbsOffset + ne*2
	if end > uint64(len(m.data)) {
		return nil, fmt.Errorf("gguf: tensor %q OOB", name)
	}
	ptr := unsafe.Pointer(&m.data[t.AbsOffset])
	return unsafe.Slice((*uint16)(ptr), ne), nil
}

// parse reads the GGUF header, metadata, and tensor directory.
func (m *GGUFModel) parse() error {
	if len(m.data) < 32 {
		return errors.New("gguf: too small")
	}
	c := cursor{data: m.data}

	// Header: magic(u32) + version(u32) + n_tensors(u64) + n_kv(u64)
	magic := c.u32()
	if magic != ggufMagic {
		return fmt.Errorf("gguf: bad magic 0x%08x", magic)
	}
	version := c.u32()
	if version != ggufVersion {
		return fmt.Errorf("gguf: version %d, want %d", version, ggufVersion)
	}
	nTensors := c.u64()
	nKV := c.u64()
	if c.err != nil {
		return c.err
	}

	// Parse metadata KV table
	m.Meta = make(map[string]interface{}, nKV)
	for i := uint64(0); i < nKV; i++ {
		key := c.str()
		vtype := c.u32()
		val := c.readValue(vtype)
		if c.err != nil {
			return fmt.Errorf("gguf: metadata entry %d: %w", i, c.err)
		}
		m.Meta[key] = val
	}

	// Parse tensor directory
	m.Tensors = make(map[string]*GGUFTensor, nTensors)
	tensors := make([]GGUFTensor, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		t := &tensors[i]
		t.Name = c.str()
		t.NDim = c.u32()
		for d := uint32(0); d < t.NDim; d++ {
			t.Dims[d] = c.u64()
		}
		t.Type = c.u32()
		relOffset := c.u64() // relative to data section start
		t.AbsOffset = relOffset // will be fixed up below
		if c.err != nil {
			return fmt.Errorf("gguf: tensor %d: %w", i, c.err)
		}
		m.Tensors[t.Name] = t
	}

	// Data section starts at current cursor position, aligned to 64 bytes
	dataStart := (c.pos + 63) & ^uint64(63)

	// Fix up relative offsets → absolute
	for i := range tensors {
		tensors[i].AbsOffset += dataStart
	}

	return nil
}

// cursor reads little-endian values from a byte slice.
type cursor struct {
	data []byte
	pos  uint64
	err  error
}

func (c *cursor) u8() uint8 {
	if c.err != nil || c.pos >= uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return 0
	}
	v := c.data[c.pos]
	c.pos++
	return v
}

func (c *cursor) u16() uint16 {
	if c.err != nil || c.pos+2 > uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint16(c.data[c.pos:])
	c.pos += 2
	return v
}

func (c *cursor) u32() uint32 {
	if c.err != nil || c.pos+4 > uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint32(c.data[c.pos:])
	c.pos += 4
	return v
}

func (c *cursor) u64() uint64 {
	if c.err != nil || c.pos+8 > uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return 0
	}
	v := binary.LittleEndian.Uint64(c.data[c.pos:])
	c.pos += 8
	return v
}

func (c *cursor) i32() int32 {
	return int32(c.u32())
}

func (c *cursor) f32() float32 {
	bits := c.u32()
	return *(*float32)(unsafe.Pointer(&bits))
}

func (c *cursor) str() string {
	n := c.u64()
	if c.err != nil || c.pos+n > uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return ""
	}
	s := string(c.data[c.pos : c.pos+n])
	c.pos += n
	return s
}

func (c *cursor) skip(n uint64) {
	if c.err != nil || c.pos+n > uint64(len(c.data)) {
		c.err = io.ErrUnexpectedEOF
		return
	}
	c.pos += n
}

// readValue reads a GGUF typed value. We keep strings and numbers;
// arrays are stored as []interface{}.
func (c *cursor) readValue(vtype uint32) interface{} {
	switch vtype {
	case ggufUint8:
		return c.u8()
	case ggufInt8:
		return int8(c.u8())
	case ggufUint16:
		return c.u16()
	case ggufInt16:
		return int16(c.u16())
	case ggufUint32:
		return c.u32()
	case ggufInt32:
		return c.i32()
	case ggufFloat32:
		return c.f32()
	case ggufBool:
		return c.u8() != 0
	case ggufString:
		return c.str()
	case ggufArray:
		elemType := c.u32()
		n := c.u64()
		if c.err != nil {
			return nil
		}
		arr := make([]interface{}, n)
		for i := uint64(0); i < n; i++ {
			arr[i] = c.readValue(elemType)
		}
		return arr
	case ggufUint64:
		return c.u64()
	case ggufInt64:
		return int64(c.u64())
	case ggufFloat64:
		bits := c.u64()
		return *(*float64)(unsafe.Pointer(&bits))
	default:
		c.err = fmt.Errorf("unknown GGUF value type %d", vtype)
		return nil
	}
}

// MetaU32 returns a uint32 metadata value.
func (m *GGUFModel) MetaU32(key string) (uint32, bool) {
	v, ok := m.Meta[key]
	if !ok {
		return 0, false
	}
	switch x := v.(type) {
	case uint32:
		return x, true
	case int32:
		return uint32(x), true
	default:
		return 0, false
	}
}

// MetaStr returns a string metadata value.
func (m *GGUFModel) MetaStr(key string) (string, bool) {
	v, ok := m.Meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// MetaStrArray returns a string array metadata value.
func (m *GGUFModel) MetaStrArray(key string) ([]string, bool) {
	v, ok := m.Meta[key]
	if !ok {
		return nil, false
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	strs := make([]string, len(arr))
	for i, a := range arr {
		s, ok := a.(string)
		if !ok {
			return nil, false
		}
		strs[i] = s
	}
	return strs, true
}
