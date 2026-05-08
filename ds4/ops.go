package ds4

import (
	"math"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// rmsNorm computes RMSNorm(x, w) in-place: x = w * x / rms(x).
func rmsNorm(x, w []float32) {
	simd.RMSNorm(x, w, RMSEps)
}

// rmsNormNoScale computes RMSNorm without learned scale.
func rmsNormNoScale(x []float32) {
	simd.RMSNormNoScale(x, RMSEps)
}

// softmax computes in-place softmax over x.
func softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := float32(0)
	for i, v := range x {
		e := float32(math.Exp(float64(v - maxVal)))
		x[i] = e
		sum += e
	}
	inv := 1.0 / sum
	for i := range x {
		x[i] *= inv
	}
}

// sigmoid computes stable sigmoid: 1/(1+exp(-x)).
func sigmoid(x float32) float32 {
	if x >= 0 {
		e := float32(math.Exp(float64(-x)))
		return 1.0 / (1.0 + e)
	}
	e := float32(math.Exp(float64(x)))
	return e / (1.0 + e)
}

// swiGLU computes dst = silu(a) * b in-place.
// silu(x) = x * sigmoid(x).
func swiGLU(dst, a, b []float32) {
	simd.VecSiLUMul(dst, a, b)
}

// ropeYaRNTailInplace applies RoPE with YaRN extrapolation to only
// the tail NRot dimensions of each head. The first NHeadDim-NRot dims
// are untouched ("NoPE" portion of MLA).
//
// q is [nHead, headDim], pos is the position, freqBase/freqScale set the
// frequency, and inverse=true de-rotates (for output projection).
func ropeYaRNTailInplace(q []float32, pos int, nHead, headDim, nRot int,
	freqBase, freqScale float32, inverse bool) {

	// YaRN correction dimensions
	loFreq := float64(RoPEYarnBetaSlow)
	hiFreq := float64(RoPEYarnBetaFast)
	origCtx := float64(RoPEOrigCtx)
	scale := float64(freqScale)

	corrLo := math.Max(math.Log(origCtx/(loFreq*2*math.Pi))/(2*math.Log(float64(freqBase))), 0)
	corrHi := math.Min(math.Log(origCtx/(hiFreq*2*math.Pi))/(2*math.Log(float64(freqBase))), float64(nRot/2-1))

	for h := 0; h < nHead; h++ {
		hOff := h * headDim
		tailOff := hOff + (headDim - nRot) // NoPE dims at the front

		for i := 0; i < nRot/2; i++ {
			// Frequency for this dimension pair
			dimF := float64(i)
			rampMix := math.Min(math.Max((dimF-corrLo)/(corrHi-corrLo), 0), 1)
			extFactor := 1.0 - rampMix // interpolation factor

			freq := 1.0 / (math.Pow(float64(freqBase), 2.0*dimF/float64(nRot)) * (extFactor*scale + (1-extFactor)))
			theta := freq * float64(pos)

			cosT := float32(math.Cos(theta))
			sinT := float32(math.Sin(theta))
			if inverse {
				sinT = -sinT
			}

			idx0 := tailOff + i
			idx1 := tailOff + nRot/2 + i
			v0 := q[idx0]
			v1 := q[idx1]
			q[idx0] = v0*cosT - v1*sinT
			q[idx1] = v0*sinT + v1*cosT
		}
	}
}

// FP8 E4M3FN quantization for KV cache compression.
// dsv4_e4m3fn_value_cpu from ds4.c: maps index 0-255 to float value.

var e4m3fnTable [256]float32

func init() {
	for i := 0; i < 256; i++ {
		sign := float32(1)
		if i&0x80 != 0 {
			sign = -1
		}
		exp := (i >> 3) & 0xF
		mant := i & 0x7
		if exp == 0 {
			// subnormal
			e4m3fnTable[i] = sign * float32(mant) * float32(math.Pow(2, -9))
		} else if exp == 15 && mant == 7 {
			e4m3fnTable[i] = float32(math.NaN())
		} else {
			e4m3fnTable[i] = sign * (1 + float32(mant)/8) * float32(math.Pow(2, float64(exp-7)))
		}
	}
}

// fp8KVQuantizeInplace quantizes the non-RoPE portion of a KV row to FP8 in-place.
// The RoPE tail (last nRot floats) is left as FP32.
func fp8KVQuantizeInplace(x []float32, headDim, nRot int) {
	nopeLen := headDim - nRot
	// Find absmax of NoPE portion
	amax := float32(0)
	for i := 0; i < nopeLen; i++ {
		v := x[i]
		if v < 0 {
			v = -v
		}
		if v > amax {
			amax = v
		}
	}
	if amax == 0 {
		return
	}
	// Scale to fit in E4M3FN range (max representable: 448)
	scale := amax / 448.0
	invScale := 1.0 / scale

	// Quantize each value: find nearest E4M3FN, then dequantize × scale
	for i := 0; i < nopeLen; i++ {
		v := x[i] * invScale
		// Binary search for nearest E4M3FN value
		ax := v
		if ax < 0 {
			ax = -ax
		}
		signBit := 0
		if v < 0 {
			signBit = 128
		}
		// Simple linear search (256 entries, fast enough)
		best := 0
		bestDiff := float32(1e30)
		for j := 0; j < 128; j++ {
			diff := e4m3fnTable[j] - ax
			if diff < 0 {
				diff = -diff
			}
			if diff < bestDiff {
				bestDiff = diff
				best = j
			}
		}
		x[i] = e4m3fnTable[signBit|best] * scale
	}
}

// matvecF16 computes out[outDim] = F16_weight[outDim, inDim] · x[inDim].
// Weight data is F16 (uint16), x and out are float32.
// Parallelized across rows.
func matvecF16(out []float32, wF16 []byte, x []float32, inDim, outDim int) {
	wU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&wF16[0])), outDim*inDim)
	parallelFor(outDim, func(start, end int) {
		for o := start; o < end; o++ {
			row := wU16[o*inDim : o*inDim+inDim]
			out[o] = DotF16(row, x)
		}
	})
}

// matvecQ8_0 computes out[outDim] = Q8_0_weight[outDim, inDim] · x[inDim].
// Parallelized across rows.
func matvecQ8_0(out []float32, wQ8 []byte, x []float32, inDim, outDim int) {
	rowBytes := (inDim / 32) * BlockQ8_0Size
	parallelFor(outDim, func(start, end int) {
		for o := start; o < end; o++ {
			row := wQ8[o*rowBytes : (o+1)*rowBytes]
			out[o] = DotQ8_0F32(row, x, inDim)
		}
	})
}

// matvecQ8_0Grouped computes grouped output projection.
// Weight is [outDim, inDim] Q8_0, but processed in groups of NOutGroup=8.
func matvecQ8_0Grouped(out []float32, wQ8 []byte, x []float32, inDim, outDim, groupSize int) {
	rowBytes := (inDim / 32) * BlockQ8_0Size
	nGroups := outDim / groupSize
	parallelFor(nGroups, func(gStart, gEnd int) {
		for g := gStart; g < gEnd; g++ {
			for j := 0; j < groupSize; j++ {
				o := g*groupSize + j
				row := wQ8[o*rowBytes : (o+1)*rowBytes]
				out[o] = DotQ8_0F32(row, x, inDim)
			}
		}
	})
}

// parallelFor splits work across goroutines. Falls back to serial for small n.
func parallelFor(n int, fn func(start, end int)) {
	if n <= 64 {
		fn(0, n)
		return
	}
	nWorkers := 8 // fixed, matches ds4.c default
	if nWorkers > n {
		nWorkers = n
	}
	var wg sync.WaitGroup
	rowsPer := (n + nWorkers - 1) / nWorkers
	for w := 0; w < nWorkers; w++ {
		start := w * rowsPer
		end := start + rowsPer
		if end > n {
			end = n
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			fn(s, e)
		}(start, end)
	}
	wg.Wait()
}
