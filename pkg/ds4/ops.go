package ds4

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-ds4/pkg/ds4/internal/simd"
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

func softplusStable(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
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

	// YaRN correction dimensions (matching ds4.c exactly)
	thetaScale := math.Pow(float64(freqBase), -2.0/float64(nRot))
	sinSign := 1.0
	if inverse {
		sinSign = -1.0
	}

	// Compute correction dimensions for YaRN ramp
	var corrDims [2]float64
	extFactorBase := 0.0
	if freqScale != 1.0 {
		// ext_factor from attn_factor: 1.0 for non-identity freq_scale
		extFactorBase = 1.0
		corrDims[0] = math.Max(ropeYaRNCorrDim(nRot, RoPEOrigCtx, float64(freqBase), float64(RoPEYarnBetaFast)), 0)
		corrDims[1] = math.Min(ropeYaRNCorrDim(nRot, RoPEOrigCtx, float64(freqBase), float64(RoPEYarnBetaSlow)), float64(nRot/2-1))
	}

	for h := 0; h < nHead; h++ {
		tail := h*headDim + (headDim - nRot) // offset to rope dims
		thetaExtrap := float64(pos)

		for i := 0; i < nRot; i += 2 {
			thetaInterp := float64(freqScale) * thetaExtrap
			theta := thetaInterp
			mscale := 1.0 // attn_factor default

			if extFactorBase != 0.0 {
				rampMix := ropeYaRNRamp(corrDims[0], corrDims[1], i) * extFactorBase
				theta = thetaInterp*(1.0-rampMix) + thetaExtrap*rampMix
				mscale *= 1.0 + 0.1*math.Log(1.0/float64(freqScale))
			}

			c := float32(math.Cos(theta) * mscale)
			s := float32(sinSign * math.Sin(theta) * mscale)

			x0 := q[tail+i+0]
			x1 := q[tail+i+1]
			q[tail+i+0] = x0*c - x1*s
			q[tail+i+1] = x0*s + x1*c

			thetaExtrap *= thetaScale
		}
	}
}

func ropeYaRNRamp(low, high float64, i int) float64 {
	y := (float64(i/2) - low) / math.Max(0.001, high-low)
	return 1.0 - math.Min(1.0, math.Max(0.0, y))
}

func ropeYaRNCorrDim(nRot int, origCtx uint64, base, beta float64) float64 {
	return float64(nRot) * math.Log(float64(origCtx)/(beta*2*math.Pi)) / (2 * math.Log(base))
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
// Matches ds4.c dsv4_fp8_kv_quantize_row_inplace_cpu exactly: 64-wide
// per-block amax, power-of-two scale, nearest-even E4M3FN round trip.
func fp8KVQuantizeInplace(x []float32, headDim, nRot int) {
	nopeLen := headDim - nRot
	for off := 0; off < nopeLen; off += 64 {
		amax := float32(0)
		for i := 0; i < 64 && off+i < nopeLen; i++ {
			av := x[off+i]
			if av < 0 {
				av = -av
			}
			if av > amax {
				amax = av
			}
		}
		if amax < 1.0e-4 {
			amax = 1.0e-4
		}
		scale := float32(math.Ldexp(1.0, int(math.Ceil(math.Log2(float64(amax/448.0))))))
		invScale := 1.0 / scale
		for i := 0; i < 64 && off+i < nopeLen; i++ {
			v := x[off+i] * invScale
			if v > 448 {
				v = 448
			} else if v < -448 {
				v = -448
			}
			x[off+i] = e4m3fnDequant(v) * scale
		}
	}
}

func e4m3fnValue(i int) float32 {
	expScale := [16]float32{0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256}
	exp := (i >> 3) & 0x0f
	mant := i & 0x07
	if exp == 0 {
		return float32(mant) * 0.001953125
	}
	return (1.0 + float32(mant)*0.125) * expScale[exp]
}

func e4m3fnDequant(x float32) float32 {
	sign := float32(1)
	ax := x
	if ax < 0 {
		sign = -1
		ax = -ax
	}
	if ax > 448 {
		ax = 448
	}
	lo, hi := 0, 126
	for lo < hi {
		mid := (lo + hi + 1) >> 1
		if e4m3fnValue(mid) <= ax {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	best := lo
	if best < 126 {
		bestDiff := ax - e4m3fnValue(best)
		if bestDiff < 0 {
			bestDiff = -bestDiff
		}
		nextDiff := ax - e4m3fnValue(best+1)
		if nextDiff < 0 {
			nextDiff = -nextDiff
		}
		if nextDiff < bestDiff || (nextDiff == bestDiff && ((best+1)&1) == 0 && (best&1) != 0) {
			best++
		}
	}
	return sign * e4m3fnValue(best)
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

// quantizeQ8_0Activation quantizes activation to per-block int8 with f32 scales.
// Matches ds4.c quantize_q8_0_activation (32-wide blocks, lrintf-style
// round-to-nearest-even under the default FP environment, clamp to int8).
func quantizeQ8_0Activation(x []float32, xq []int8, xscale []float32, xsum []float32) {
	nBlocks := (len(x) + 31) / 32
	for b := 0; b < nBlocks; b++ {
		off := b * 32
		bn := len(x) - off
		if bn > 32 {
			bn = 32
		}
		amax := float32(0)
		for i := 0; i < bn; i++ {
			v := x[off+i]
			if v < 0 {
				v = -v
			}
			if v > amax {
				amax = v
			}
		}
		d := amax / 127.0
		xscale[b] = d
		id := float32(0)
		if d != 0 {
			id = 1.0 / d
		}
		sum := int32(0)
		for i := 0; i < bn; i++ {
			v := int(math.RoundToEven(float64(x[off+i] * id)))
			if v > 127 {
				v = 127
			} else if v < -128 {
				v = -128
			}
			xq[off+i] = int8(v)
			sum += int32(int8(v))
		}
		for i := bn; i < 32 && off+i < len(xq); i++ {
			xq[off+i] = 0
		}
		if len(xsum) > b {
			xsum[b] = float32(sum)
		}
	}
}

func dotQ8_0Prequant(row []byte, xq []int8, xscale []float32, xsum []float32, nBlocks int) float32 {
	// Use the straightforward C-equivalent loop. The older amd64 asm helper is
	// currently kept only for vet/link compatibility; this path is the reference
	// for correctness.
	_ = xsum
	acc := float32(0)
	for b := 0; b < nBlocks; b++ {
		off := b * BlockQ8_0Size
		d := F16ToF32(*(*uint16)(unsafe.Pointer(&row[off])))
		dot := int32(0)
		for i := 0; i < 32; i++ {
			dot += int32(int8(row[off+2+i])) * int32(xq[b*32+i])
		}
		acc += d * xscale[b] * float32(dot)
	}
	return acc
}

// matvecQ8_0 computes out[outDim] = Q8_0_weight[outDim, inDim] · x[inDim].
// Uses SIMD DotQ8_0F32 (VCVTPH2PS+VPMOVSXBD+VCVTDQ2PS+FMA) directly per row.
func matvecQ8_0(out []float32, wQ8 []byte, x []float32, inDim, outDim int) {
	nBlocks := (inDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size
	xq := make([]int8, nBlocks*32)
	xscale := make([]float32, nBlocks)
	xsum := make([]float32, nBlocks)
	quantizeQ8_0Activation(x[:inDim], xq, xscale, xsum)
	parallelFor(outDim, func(start, end int) {
		for o := start; o < end; o++ {
			row := wQ8[o*rowBytes : (o+1)*rowBytes]
			out[o] = dotQ8_0Prequant(row, xq, xscale, xsum, nBlocks)
		}
	})
}

func strictGPUEngine(engine interface{}) (*Engine, bool) {
	e, ok := engine.(*Engine)
	return e, ok && e.StrictGPU
}

// matvecQ8_0GPU tries GPU dispatch first, falls back to CPU unless strict GPU is enabled.
func matvecQ8_0GPU(out []float32, wQ8 []byte, x []float32, inDim, outDim int, engine interface{}, tensorName string) {
	if engine != nil {
		if e, ok := engine.(*Engine); ok {
			if e.gpuMatvecQ8_0(out, tensorName, x, inDim, outDim) {
				return
			}
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: Q8_0 matvec fallback refused for %s [%d->%d]", tensorName, inDim, outDim))
			}
		}
	}
	matvecQ8_0(out, wQ8, x, inDim, outDim)
}

// matvecQ8_0GPULayer is like matvecQ8_0GPU but constructs the tensor name from layer + suffix.
func matvecQ8_0GPULayer(out []float32, wQ8 []byte, x []float32, inDim, outDim int, ds *DecodeState, suffix string) {
	if ds.Engine != nil {
		name := fmt.Sprintf("blk.%d.%s", ds.LayerIdx, suffix)
		if e, ok := ds.Engine.(*Engine); ok {
			if e.gpuMatvecQ8_0(out, name, x, inDim, outDim) {
				return
			}
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: Q8_0 layer matvec fallback refused for %s [%d->%d]", name, inDim, outDim))
			}
		}
	}
	matvecQ8_0(out, wQ8, x, inDim, outDim)
}

// matvecQ8_0Grouped computes grouped output projection.
func matvecQ8_0Grouped(out []float32, wQ8 []byte, x []float32, inDim, outDim, groupSize int, ds *DecodeState, suffix string) {
	if ds != nil && ds.Engine != nil {
		name := fmt.Sprintf("blk.%d.%s", ds.LayerIdx, suffix)
		if e, ok := ds.Engine.(*Engine); ok {
			if e.gpuMatvecQ8_0Grouped(out, name, x, inDim, outDim, groupSize) {
				return
			}
			if e.StrictGPU {
				panic(fmt.Sprintf("strict GPU: grouped Q8_0 fallback refused for %s [%d->%d groups=%d]", name, inDim, outDim, groupSize))
			}
		}
	}
	// C parity: attn_output_a is laid out as nGroups contiguous rank-row
	// matrices, each consuming that group's slice of heads. inDim is the total
	// head width; outDim is the per-group rank; groupSize is nGroups.
	nGroups := groupSize
	groupDim := inDim / nGroups
	rank := outDim
	nBlocks := (groupDim + 31) / 32
	rowBytes := nBlocks * BlockQ8_0Size
	xq := make([]int8, nGroups*nBlocks*32)
	xscale := make([]float32, nGroups*nBlocks)
	xsum := make([]float32, nGroups*nBlocks)
	for g := 0; g < nGroups; g++ {
		quantizeQ8_0Activation(
			x[g*groupDim:(g+1)*groupDim],
			xq[g*nBlocks*32:(g+1)*nBlocks*32],
			xscale[g*nBlocks:(g+1)*nBlocks],
			xsum[g*nBlocks:(g+1)*nBlocks],
		)
	}
	parallelFor(nGroups*rank, func(start, end int) {
		for idx := start; idx < end; idx++ {
			g := idx / rank
			gxq := xq[g*nBlocks*32 : (g+1)*nBlocks*32]
			gscale := xscale[g*nBlocks : (g+1)*nBlocks]
			gsum := xsum[g*nBlocks : (g+1)*nBlocks]
			row := wQ8[idx*rowBytes : (idx+1)*rowBytes]
			out[idx] = dotQ8_0Prequant(row, gxq, gscale, gsum, nBlocks)
		}
	})
}

// parallelFor splits work across goroutines. Falls back to serial for small n.
func parallelFor(n int, fn func(start, end int)) {
	if n <= 32 {
		fn(0, n)
		return
	}
	nWorkers := runtime.NumCPU()
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

// matvecAuto detects the quantization format from tensor size and dispatches.
func matvecAuto(out []float32, w []byte, x []float32, inDim, outDim int) {
	if len(w) == 0 {
		return
	}
	bytesPerRow := len(w) / outDim
	q8RowBytes := (inDim / 32) * BlockQ8_0Size
	q3kRowBytes := ((inDim + QK_K - 1) / QK_K) * BlockQ3KSize
	q2kRowBytes := ((inDim + QK_K - 1) / QK_K) * BlockQ2KSize

	switch {
	case bytesPerRow == q8RowBytes:
		// Q8_0
		matvecQ8_0(out, w, x, inDim, outDim)
	case bytesPerRow == q3kRowBytes:
		// Q3_K — need Q8_K quantized activation
		xQ8K := make([]byte, ((inDim+QK_K-1)/QK_K)*BlockQ8KSize)
		padded := make([]float32, ((inDim+QK_K-1)/QK_K)*QK_K)
		copy(padded, x)
		QuantizeRowQ8K(padded, xQ8K)
		parallelFor(outDim, func(start, end int) {
			for o := start; o < end; o++ {
				row := w[o*bytesPerRow : (o+1)*bytesPerRow]
				out[o] = VecDotQ3KQ8K(inDim, row, xQ8K)
			}
		})
	case bytesPerRow == q2kRowBytes:
		// Q2_K
		xQ8K := make([]byte, ((inDim+QK_K-1)/QK_K)*BlockQ8KSize)
		padded := make([]float32, ((inDim+QK_K-1)/QK_K)*QK_K)
		copy(padded, x)
		QuantizeRowQ8K(padded, xQ8K)
		parallelFor(outDim, func(start, end int) {
			for o := start; o < end; o++ {
				row := w[o*bytesPerRow : (o+1)*bytesPerRow]
				out[o] = VecDotQ2KQ8K(inDim, row, xQ8K)
			}
		})
	default:
		// Try Q6_K
		q6kRowBytes := ((inDim + QK_K - 1) / QK_K) * 210
		if bytesPerRow == q6kRowBytes {
			xQ8K := make([]byte, ((inDim+QK_K-1)/QK_K)*BlockQ8KSize)
			padded := make([]float32, ((inDim+QK_K-1)/QK_K)*QK_K)
			copy(padded, x)
			QuantizeRowQ8K(padded, xQ8K)
			parallelFor(outDim, func(start, end int) {
				for o := start; o < end; o++ {
					row := w[o*bytesPerRow : (o+1)*bytesPerRow]
					out[o] = VecDotQ6KQ8K(inDim, row, xQ8K)
				}
			})
			return
		}
		// Try IQ4_NL (use Q8K-quantized activation for int8 dot)
		iq4RowBytes := ((inDim + QK4_NL - 1) / QK4_NL) * BlockIQ4NLSize
		if bytesPerRow == iq4RowBytes {
			xQ8K := make([]byte, ((inDim+QK_K-1)/QK_K)*BlockQ8KSize)
			padded := make([]float32, ((inDim+QK_K-1)/QK_K)*QK_K)
			copy(padded, x)
			QuantizeRowQ8K(padded, xQ8K)
			parallelFor(outDim, func(start, end int) {
				for o := start; o < end; o++ {
					row := w[o*bytesPerRow : (o+1)*bytesPerRow]
					out[o] = VecDotIQ4NLQ8K(inDim, row, xQ8K)
				}
			})
			return
		}
		for i := range out[:outDim] {
			out[i] = 0
		}
	}
}

func hasNaNF32(x []float32) bool {
	for _, v := range x {
		if v != v {
			return true
		} // NaN != NaN
	}
	return false
}

// detectOutDim tries all known quant formats to determine the output dimension
// of a weight tensor given its byte length and input dimension.
func detectOutDim(w []byte, inDim int) int {
	totalBytes := len(w)
	candidates := []int{
		(inDim / 32) * BlockQ8_0Size,                     // Q8_0: 34 bytes per 32 elements
		((inDim + QK_K - 1) / QK_K) * BlockQ2KSize,       // Q2_K: 84 bytes per 256 elements
		((inDim + QK_K - 1) / QK_K) * 110,                // Q3_K: 110 bytes per 256 elements
		((inDim + QK_K - 1) / QK_K) * 210,                // Q6_K: 210 bytes per 256 elements
		((inDim + QK4_NL - 1) / QK4_NL) * BlockIQ4NLSize, // IQ4_NL: 18 bytes per 32 elements
	}
	for _, rowBytes := range candidates {
		if rowBytes > 0 && totalBytes%rowBytes == 0 {
			return totalBytes / rowBytes
		}
	}
	return 0
}
