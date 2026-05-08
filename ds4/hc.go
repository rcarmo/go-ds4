package ds4

import (
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

	// 3. Apply scale + base + sigmoid
	scale := unsafe.Slice((*float32)(unsafe.Pointer(&scaleTensor[0])), 3)
	base := unsafe.Slice((*float32)(unsafe.Pointer(&baseTensor[0])), hcMixDim)

	for i := 0; i < hcMixDim; i++ {
		var s float32
		if i < NHC {
			s = scale[0]
		} else if i < 2*NHC {
			s = scale[1]
		} else {
			s = scale[2]
		}
		mix[i] = sigmoid(s*mix[i]) + base[i]
	}

	// Extract pre, post, comb from mix
	pre := mix[0:NHC]
	copy(post, mix[NHC:2*NHC])
	copy(comb, mix[2*NHC:2*NHC+NHC*NHC])

	// Clamp pre and post
	for i := range pre {
		if pre[i] < HCEps {
			pre[i] = HCEps
		}
	}
	for i := range post {
		if post[i] < HCEps {
			post[i] = HCEps
		}
	}

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
//	outHC[s] = post[s] * blockOut + Σ_t (comb[s*NHC+t] * residualHC[t])
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
				val += comb[s*NHC+t] * residualHC[t*NEmbd+i]
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
