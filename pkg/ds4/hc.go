package ds4

import (
	"math"
	"unsafe"
)

// hcDim is the total HC state size: NHC * NEmbd = 4 * 4096 = 16384.
const hcDim = NHC * NEmbd

// hcMixDim is the HC mixing parameter count: 2*NHC + NHC² = 8 + 16 = 24.
const hcMixDim = 2*NHC + NHC*NHC

// hcPreFromState computes the HC pre-step:
//  1. Flatten HC state [NHC, NEmbd] → RMSNorm
//  2. Project through fn (F16) → mix params [hcMixDim]
//  3. Sigmoid + scale → pre[NHC], post[NHC], comb[NHC²]
//  4. out = Σ_s (pre[s] * state[s])
//
// fn: F16 [hcDim, hcMixDim], scale: F32 [3], base: F32 [hcMixDim]
func hcPreFromState(
	out []float32, // [NEmbd] — input to sublayer
	post []float32, // [NHC] — saved for hcPost
	comb []float32, // [NHC*NHC] — saved for hcPost
	residualHC []float32, // [NHC, NEmbd] — current HC state
	fn []byte, // F16 weights
	scaleTensor []byte, // F32 [3]
	baseTensor []byte, // F32 [hcMixDim]
	flatScratch []float32, // [hcDim]
	mixScratch []float32, // [hcMixDim]
) {
	// 1. Flatten + RMSNorm
	flat := flatScratch[:hcDim]
	copy(flat, residualHC[:hcDim])
	rmsNormNoScale(flat)

	// 2. Project: mix[hcMixDim] = flat · fn^T
	mix := mixScratch[:hcMixDim]
	fnU16 := unsafe.Slice((*uint16)(unsafe.Pointer(&fn[0])), hcDim*hcMixDim)
	for j := 0; j < hcMixDim; j++ {
		row := fnU16[j*hcDim : (j+1)*hcDim]
		mix[j] = DotF16(row, flat)
	}

	// 3. Sinkhorn split (matches ds4.c hc_split_sinkhorn_one)
	scale := unsafe.Slice((*float32)(unsafe.Pointer(&scaleTensor[0])), 3)
	base := unsafe.Slice((*float32)(unsafe.Pointer(&baseTensor[0])), hcMixDim)
	preScale, postScale, combScale := scale[0], scale[1], scale[2]
	eps := float32(HCEps)

	pre := mix[0:NHC]
	for i := 0; i < NHC; i++ {
		z := mix[i]*preScale + base[i]
		pre[i] = sigmoid(z) + eps
	}
	for i := 0; i < NHC; i++ {
		off := NHC + i
		z := mix[off]*postScale + base[off]
		post[i] = 2 * sigmoid(z)
	}

	// comb matrix c[src + dst*NHC]
	var c [NHC * NHC]float32
	for dst := 0; dst < NHC; dst++ {
		rowMax := float32(-1e30)
		for src := 0; src < NHC; src++ {
			idx := src + dst*NHC
			off := 2*NHC + idx
			v := mix[off]*combScale + base[off]
			c[idx] = v
			if v > rowMax {
				rowMax = v
			}
		}
		rowSum := float32(0)
		for src := 0; src < NHC; src++ {
			idx := src + dst*NHC
			v := float32(math.Exp(float64(c[idx] - rowMax)))
			c[idx] = v
			rowSum += v
		}
		inv := float32(1.0) / rowSum
		for src := 0; src < NHC; src++ {
			idx := src + dst*NHC
			c[idx] = c[idx]*inv + eps
		}
	}
	for src := 0; src < NHC; src++ {
		sum := float32(0)
		for dst := 0; dst < NHC; dst++ {
			sum += c[src+dst*NHC]
		}
		inv := float32(1.0) / (sum + eps)
		for dst := 0; dst < NHC; dst++ {
			c[src+dst*NHC] *= inv
		}
	}
	for iter := 1; iter < NHCSinkhornIter; iter++ {
		for dst := 0; dst < NHC; dst++ {
			sum := float32(0)
			for src := 0; src < NHC; src++ {
				sum += c[src+dst*NHC]
			}
			inv := float32(1.0) / (sum + eps)
			for src := 0; src < NHC; src++ {
				c[src+dst*NHC] *= inv
			}
		}
		for src := 0; src < NHC; src++ {
			sum := float32(0)
			for dst := 0; dst < NHC; dst++ {
				sum += c[src+dst*NHC]
			}
			inv := float32(1.0) / (sum + eps)
			for dst := 0; dst < NHC; dst++ {
				c[src+dst*NHC] *= inv
			}
		}
	}
	copy(comb, c[:])

	// 4. out = Σ_s (pre[s] * residualHC[s*NEmbd : (s+1)*NEmbd])
	for i := range out[:NEmbd] {
		out[i] = 0
	}
	for s := 0; s < NHC; s++ {
		p := pre[s]
		for i := 0; i < NEmbd; i++ {
			out[i] += p * residualHC[s*NEmbd+i]
		}
	}
}

// hcPostOne injects a sublayer output into the HC state and mixes streams.
//
//	outHC[dst] = post[dst] * blockOut + Σ_src (comb[dst + src*NHC] * residualHC[src])
func hcPostOne(
	outHC []float32, // [NHC, NEmbd] — updated HC state
	blockOut []float32, // [NEmbd] — sublayer output
	residualHC []float32, // [NHC, NEmbd] — previous HC state
	post []float32, // [NHC]
	comb []float32, // [NHC*NHC]
) {
	for s := 0; s < NHC; s++ {
		off := s * NEmbd
		for i := 0; i < NEmbd; i++ {
			val := post[s] * blockOut[i]
			for t := 0; t < NHC; t++ {
				// C reference addresses HC combine as [dst, src] using comb[dst + src*NHC].
				val += comb[s+t*NHC] * residualHC[t*NEmbd+i]
			}
			outHC[off+i] = val
		}
	}
}

// hcPostSumOne is like hcPostOne but adds two sublayer outputs
// (routed expert + shared expert) before HC mixing.
func hcPostSumOne(
	outHC []float32,
	routedOut, sharedOut []float32,
	residualHC []float32,
	post, comb []float32,
	tmp []float32,
) {
	t := tmp[:NEmbd]
	for i := 0; i < NEmbd; i++ {
		t[i] = routedOut[i] + sharedOut[i]
	}
	hcPostOne(outHC, t, residualHC, post, comb)
}
