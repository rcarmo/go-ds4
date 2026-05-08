package ds4

import "math"

func layerAttnDecodeV2(ds *DecodeState, layer *LayerWeights, cache *LayerCache, pos, il int) {
	cfg := ds.Cfg()
	nHead := cfg.NHead
	qkNope := 128
	qkRope := cfg.NRot
	qHeadDim := qkNope + qkRope
	vHeadDim := cfg.NValueDim
	kvLoraRank := 512
	kvBHeadDim := qkNope + vHeadDim

	if eng, ok := ds.Engine.(*Engine); ok {
		if v, ok2 := eng.Model.MetaU32("deepseek2.attention.kv_lora_rank"); ok2 {
			kvLoraRank = int(v)
		}
	}

	normW := tensorF32Unsafe(layer.AttnNorm)
	copy(ds.AttnNormed[:cfg.NEmbd], ds.AttnCur[:cfg.NEmbd])
	rmsNorm(ds.AttnNormed[:cfg.NEmbd], normW[:cfg.NEmbd])

	qFull := make([]float32, nHead*qHeadDim)
	matvecAuto(qFull, layer.AttnQ, ds.AttnNormed[:cfg.NEmbd], cfg.NEmbd, nHead*qHeadDim)
	for h := 0; h < nHead; h++ {
		rmsNormNoScale(qFull[h*qHeadDim : (h+1)*qHeadDim])
	}

	kvADim := kvLoraRank + qkRope
	kvA := make([]float32, kvADim)
	matvecAuto(kvA, layer.AttnKVA_MQA, ds.AttnNormed[:cfg.NEmbd], cfg.NEmbd, kvADim)
	if len(layer.AttnKVANorm) > 0 {
		kvNormW := tensorF32Unsafe(layer.AttnKVANorm)
		rmsNorm(kvA[:kvLoraRank], kvNormW[:kvLoraRank])
	}

	freqBase := cfg.RoPEFreqBase
	for h := 0; h < nHead; h++ {
		ropeYaRNTailOnly(qFull[h*qHeadDim+qkNope:(h+1)*qHeadDim], pos, qkRope, freqBase, 1.0)
	}
	ropeYaRNTailOnly(kvA[kvLoraRank:kvADim], pos, qkRope, freqBase, 1.0)

	kvBFull := make([]float32, nHead*kvBHeadDim)
	matvecAuto(kvBFull, layer.AttnKVB, kvA[:kvLoraRank], kvLoraRank, nHead*kvBHeadDim)

	cacheRow := make([]float32, kvADim)
	copy(cacheRow, kvA)
	// Store in padded raw cache
	padded := make([]float32, NHeadDim)
	if kvADim <= NHeadDim {
		copy(padded, cacheRow)
	}
	cache.PushRawKV(padded)

	nRaw := cache.NRaw
	if nRaw > cache.CapRaw {
		nRaw = cache.CapRaw
	}
	scale := float32(1.0 / math.Sqrt(float64(qkNope+qkRope)))
	headsOut := make([]float32, nHead*vHeadDim)

	for h := 0; h < nHead; h++ {
		qNope := qFull[h*qHeadDim : h*qHeadDim+qkNope]
		qRope := qFull[h*qHeadDim+qkNope : (h+1)*qHeadDim]
		kNope := kvBFull[h*kvBHeadDim : h*kvBHeadDim+qkNope]
		vHead := kvBFull[h*kvBHeadDim+qkNope : h*kvBHeadDim+qkNope+vHeadDim]

		scores := make([]float32, nRaw)
		maxScore := float32(-1e30)
		for t := 0; t < nRaw; t++ {
			row := cache.RawRow(t)
			cachedRope := row[kvLoraRank : kvLoraRank+qkRope]
			dot := float32(0)
			if t == nRaw-1 {
				for d := 0; d < qkNope; d++ {
					dot += qNope[d] * kNope[d]
				}
			}
			for d := 0; d < qkRope; d++ {
				dot += qRope[d] * cachedRope[d]
			}
			scores[t] = dot * scale
			if scores[t] > maxScore {
				maxScore = scores[t]
			}
		}
		denom := float32(0)
		for t := range scores {
			scores[t] = float32(math.Exp(float64(scores[t] - maxScore)))
			denom += scores[t]
		}
		if denom > 0 {
			inv := 1.0 / denom
			for t := range scores {
				scores[t] *= inv
			}
		}
		hOut := headsOut[h*vHeadDim : (h+1)*vHeadDim]
		for d := 0; d < vHeadDim; d++ {
			hOut[d] = vHead[d]
		}
	}
	matvecAuto(ds.AttnOut[:cfg.NEmbd], layer.AttnOutput, headsOut, nHead*vHeadDim, cfg.NEmbd)
}

func ropeYaRNTailOnly(x []float32, pos int, nRot int, freqBase, freqScale float32) {
	for i := 0; i < nRot/2; i++ {
		freq := 1.0 / (math.Pow(float64(freqBase), 2.0*float64(i)/float64(nRot)) * float64(freqScale))
		theta := freq * float64(pos)
		cosT := float32(math.Cos(theta))
		sinT := float32(math.Sin(theta))
		v0 := x[i]
		v1 := x[nRot/2+i]
		x[i] = v0*cosT - v1*sinT
		x[nRot/2+i] = v0*sinT + v1*cosT
	}
}
