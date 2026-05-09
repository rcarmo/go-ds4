package ds4

import (
	"math"
	"unsafe"
)

// TurboQuant: KV cache compression using Walsh-Hadamard rotation + uniform scalar quantization.
// Based on arXiv:2504.19874.
//
// Encode: x → FWHT(x/‖x‖) → uniform b-bit quantize → packed indices + norm
// Decode: indices → dequantize → inverse FWHT → × norm → reconstructed x
//
// Storage per vector: 4 bytes (norm) + ceil(dim*bits/8) bytes (indices)

type TurboQuantConfig struct {
	Bits   int // 2, 3, or 4
	Dim    int // original dimension
	PadDim int // next power of 2
}

func NewTurboQuantConfig(dim, bits int) *TurboQuantConfig {
	return &TurboQuantConfig{Bits: bits, Dim: dim, PadDim: nextPow2(dim)}
}

// CompressedSize returns bytes per vector: 4 (norm) + ceil(dim*bits/8).
func (cfg *TurboQuantConfig) CompressedSize() int {
	return 12 + (cfg.PadDim*cfg.Bits+7)/8
}

// Encode compresses x into out. out must be CompressedSize() bytes.
func (cfg *TurboQuantConfig) Encode(x []float32, out []byte) {
	dim := cfg.Dim
	padDim := cfg.PadDim
	bits := cfg.Bits
	nLevels := 1 << bits

	// Compute norm
	norm := float32(0)
	for i := 0; i < dim; i++ {
		norm += x[i] * x[i]
	}
	norm = float32(math.Sqrt(float64(norm)))

	// Store norm
	*(*float32)(unsafe.Pointer(&out[0])) = norm
	if norm == 0 {
		for i := 4; i < len(out); i++ {
			out[i] = 0
		}
		return
	}

	// Normalize + pad
	tmp := make([]float32, padDim)
	invNorm := 1.0 / norm
	for i := 0; i < dim; i++ {
		tmp[i] = x[i] * invNorm
	}

	// FWHT
	Fwht(tmp)
	invSqrt := float32(1.0 / math.Sqrt(float64(padDim)))
	for i := range tmp {
		tmp[i] *= invSqrt
	}

	// Find actual range of transformed values
	vmin, vmax := tmp[0], tmp[0]
	for i := 1; i < padDim; i++ {
		if tmp[i] < vmin {
			vmin = tmp[i]
		}
		if tmp[i] > vmax {
			vmax = tmp[i]
		}
	}
	if vmin == vmax {
		vmax = vmin + 1
	}
	*(*float32)(unsafe.Pointer(&out[4])) = vmin
	*(*float32)(unsafe.Pointer(&out[8])) = vmax

	data := out[12:]
	for i := range data {
		data[i] = 0
	}
	bitPos := 0
	span := vmax - vmin
	if span == 0 {
		span = 1
	}
	for i := 0; i < padDim; i++ {
		q := int((tmp[i]-vmin)/span*float32(nLevels-1) + 0.5)
		if q >= nLevels {
			q = nLevels - 1
		}
		if q < 0 {
			q = 0
		}
		// Pack bits
		byteIdx := bitPos / 8
		bitOff := bitPos % 8
		data[byteIdx] |= byte(q) << bitOff
		if bitOff+bits > 8 {
			data[byteIdx+1] |= byte(q) >> (8 - bitOff)
		}
		bitPos += bits
	}
}

// Decode reconstructs from compressed representation.
func (cfg *TurboQuantConfig) Decode(compressed []byte, out []float32) {
	dim := cfg.Dim
	padDim := cfg.PadDim
	bits := cfg.Bits
	nLevels := 1 << bits
	mask := byte(nLevels - 1)

	norm := *(*float32)(unsafe.Pointer(&compressed[0]))
	if norm == 0 {
		for i := range out[:dim] {
			out[i] = 0
		}
		return
	}

	// Unpack indices → dequantize
	vmin := *(*float32)(unsafe.Pointer(&compressed[4]))
	vmax := *(*float32)(unsafe.Pointer(&compressed[8]))
	span := vmax - vmin
	data := compressed[12:]
	tmp := make([]float32, padDim)
	bitPos := 0
	for i := 0; i < padDim; i++ {
		byteIdx := bitPos / 8
		bitOff := bitPos % 8
		idx := (data[byteIdx] >> bitOff) & mask
		if bitOff+bits > 8 {
			idx |= (data[byteIdx+1] << (8 - bitOff)) & mask
		}
		// Map [0, nLevels-1] → [vmin, vmax]
		tmp[i] = float32(idx)/float32(nLevels-1)*span + vmin
		bitPos += bits
	}

	// Inverse: FWHT(FWHT(x)/√N) = FWHT(y) where y=FWHT(x)/√N
	// FWHT(y) = FWHT(FWHT(x)/√N) = FWHT(FWHT(x))/√N = N*x/√N = √N*x
	// So: x = FWHT(y) / √N
	// But we also divided x by norm at encode. So: out = FWHT(y) / √N * norm
	for i := range tmp {
		tmp[i] *= float32(math.Sqrt(float64(padDim)))
	}
	Fwht(tmp)
	scale := norm / float32(padDim)
	for i := 0; i < dim; i++ {
		out[i] = tmp[i] * scale
	}
}

// fwht performs the in-place Fast Walsh-Hadamard Transform.
func Fwht(x []float32) {
	n := len(x)
	h := 1
	for h < n {
		for i := 0; i < n; i += h * 2 {
			for j := i; j < i+h; j++ {
				a := x[j]
				b := x[j+h]
				x[j] = a + b
				x[j+h] = a - b
			}
		}
		h *= 2
	}
}

func nextPow2(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}
