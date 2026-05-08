package ds4

import (
	"math"

	"github.com/rcarmo/go-ds4/ds4/simd"
)

// compressorDecodeOne updates compressor rolling state for one token and
// emits a compressed KV row when reaching the compression boundary.
// Mirrors compressor_decode_one_decode_scratch() in ds4.c.
func compressorDecodeOne(
	outComp, kvCur, scoreCur, pooled []float32,
	wKV, wGate, wAPE, wNorm []byte,
	x []float32,
	stateKV, stateScore []float32,
	headDim, compressRatio, il, pos int,
	quantizeKV bool,
) bool {
	if compressRatio == 0 || len(wKV) == 0 || len(wGate) == 0 || len(wAPE) == 0 || len(wNorm) == 0 {
		return false
	}
	coff := 1
	if compressRatio == 4 {
		coff = 2
	}
	width := coff * headDim
	posMod := pos % compressRatio
	row := posMod
	if compressRatio == 4 {
		row = compressRatio + posMod
	}
	shouldCompress := ((pos + 1) % compressRatio) == 0

	kv := kvCur[:width]
	sc := scoreCur[:width]
	matvecQ8_0(kv, wKV, x, NEmbd, width)
	matvecQ8_0(sc, wGate, x, NEmbd, width)

	// Add APE[j, pos_mod] where tensor layout is [dim0=width, dim1=ratio]
	apeU16 := tensorU16Unsafe(wAPE)
	for j := 0; j < width; j++ {
		sc[j] += F16ToF32(apeU16[posMod*width+j])
	}

	copy(stateKV[row*width:(row+1)*width], kv)
	copy(stateScore[row*width:(row+1)*width], sc)
	if !shouldCompress {
		return false
	}

	compressorPoolDecodeState(pooled[:headDim], stateKV, stateScore, headDim, compressRatio)

	// RMSNorm + scale
	normW := tensorF32Unsafe(wNorm)
	ss := float64(0)
	for i := 0; i < headDim; i++ {
		v := pooled[i]
		ss += float64(v * v)
	}
	rms := float32(1.0 / math.Sqrt(ss/float64(headDim)+RMSEps))
	for i := 0; i < headDim; i++ {
		outComp[i] = pooled[i] * rms * normW[i]
	}

	compPos := pos + 1 - compressRatio
	ropeYaRNTailInplace(outComp[:headDim], compPos, 1, headDim, NRot, layerRoPEFreqBase(il), layerRoPEFreqScale(il), false)
	if quantizeKV && headDim == NHeadDim {
		fp8KVQuantizeInplace(outComp[:headDim], headDim, NRot)
	}

	// ratio-4 keeps two lanes; copy second half to first then mirror back
	if compressRatio == 4 {
		for r := 0; r < compressRatio; r++ {
			copy(stateKV[r*width:(r+1)*width], stateKV[(compressRatio+r)*width:(compressRatio+r+1)*width])
			copy(stateScore[r*width:(r+1)*width], stateScore[(compressRatio+r)*width:(compressRatio+r+1)*width])
		}
		for r := 0; r < compressRatio; r++ {
			copy(stateKV[(compressRatio+r)*width:(compressRatio+r+1)*width], stateKV[r*width:(r+1)*width])
			copy(stateScore[(compressRatio+r)*width:(compressRatio+r+1)*width], stateScore[r*width:(r+1)*width])
		}
	}
	return true
}

// compressorPoolDecodeState pools compressor window rows with a softmax over
// per-dimension scores. Mirrors compressor_pool_decode_state() in ds4.c.
func compressorPoolDecodeState(out, stateKV, stateScore []float32, headDim, compressRatio int) {
	coff := 1
	if compressRatio == 4 {
		coff = 2
	}
	width := coff * headDim
	negInf := float32(-1e30)

	for j := 0; j < headDim; j++ {
		maxScore := negInf

		if compressRatio == 4 {
			for r := 0; r < compressRatio; r++ {
				sp := stateScore[r*width+j]
				sc := stateScore[(compressRatio+r)*width+headDim+j]
				if sp > maxScore {
					maxScore = sp
				}
				if sc > maxScore {
					maxScore = sc
				}
			}
		} else {
			for r := 0; r < compressRatio; r++ {
				s := stateScore[r*width+j]
				if s > maxScore {
					maxScore = s
				}
			}
		}

		if maxScore <= negInf*0.5 {
			out[j] = 0
			continue
		}

		denom := float32(0)
		sum := float32(0)
		if compressRatio == 4 {
			for r := 0; r < compressRatio; r++ {
				wp := float32(math.Exp(float64(stateScore[r*width+j] - maxScore)))
				wc := float32(math.Exp(float64(stateScore[(compressRatio+r)*width+headDim+j] - maxScore)))
				denom += wp + wc
				sum += wp*stateKV[r*width+j] + wc*stateKV[(compressRatio+r)*width+headDim+j]
			}
		} else {
			for r := 0; r < compressRatio; r++ {
				w := float32(math.Exp(float64(stateScore[r*width+j] - maxScore)))
				denom += w
				sum += w * stateKV[r*width+j]
			}
		}
		if denom > 0 {
			out[j] = sum / denom
		} else {
			out[j] = 0
		}
	}
}

// indexerAllowedDecodeOne selects which compressed rows are visible for ratio-4
// layers, using the indexer auxiliary projection. Mirrors ds4.c behavior.
func indexerAllowedDecodeOne(
	ds *DecodeState,
	layer *LayerWeights,
	cur []float32,
	qrNorm []float32,
	indexComp []float32,
	nComp, il, pos int,
) []bool {
	if nComp == 0 || len(layer.IndexerQB) == 0 || len(layer.IndexerProj) == 0 {
		return nil
	}
	if nComp > len(ds.IndexAllowed) {
		nComp = len(ds.IndexAllowed)
	}
	allowed := ds.IndexAllowed[:nComp]
	for i := range allowed {
		allowed[i] = false
	}
	topK := NIndexerTopK
	if topK > nComp {
		topK = nComp
	}
	if topK == nComp {
		for i := range allowed {
			allowed[i] = true
		}
		return allowed
	}

	matvecQ8_0(ds.IndexQ, layer.IndexerQB, qrNorm, NLoraQ, NIndexerHead*NIndexerHeadDim)
	ropeYaRNTailInplace(ds.IndexQ, pos, NIndexerHead, NIndexerHeadDim, NRot, layerRoPEFreqBase(il), layerRoPEFreqScale(il), false)

	matvecQ8_0(ds.IndexWeights, layer.IndexerProj, cur, NEmbd, NIndexerHead)
	scale := float32(1.0 / math.Sqrt(float64(NIndexerHeadDim*NIndexerHead)))
	for h := 0; h < NIndexerHead; h++ {
		ds.IndexWeights[h] *= scale
	}

	scores := ds.IndexScores[:nComp]
	for c := 0; c < nComp; c++ {
		kv := indexComp[c*NIndexerHeadDim : (c+1)*NIndexerHeadDim]
		s := float32(0)
		for h := 0; h < NIndexerHead; h++ {
			qh := ds.IndexQ[h*NIndexerHeadDim : (h+1)*NIndexerHeadDim]
			dot := simd.Sdot(kv, qh)
			if dot < 0 {
				dot = 0
			}
			s += dot * ds.IndexWeights[h]
		}
		scores[c] = s
	}

	for k := 0; k < topK; k++ {
		best := -1
		bestScore := float32(-1e30)
		for c := 0; c < nComp; c++ {
			if !allowed[c] && scores[c] > bestScore {
				best = c
				bestScore = scores[c]
			}
		}
		if best >= 0 {
			allowed[best] = true
		}
	}
	return allowed
}
