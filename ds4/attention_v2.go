package ds4

import "math"

// layerAttnDecodeV2 implements DeepSeek-V2 MLA decode attention.
//
// Shapes (V2 Lite):
//
//	Q:      [nHead * (qkNope + qkRope)]
//	KV_a:   [kvLoraRank + qkRope] (compressed cache representation)
//	KV_b:   [kvLoraRank -> nHead * (qkNope + vHeadDim)]
func layerAttnDecodeV2(ds *DecodeState, layer *LayerWeights, cache *LayerCache, pos, il int) {
	cfg := ds.Cfg()
	nHead := cfg.NHead
	qkRope := cfg.NRot
	qkNope := cfg.NHeadDim - qkRope
	qHeadDim := qkNope + qkRope
	vHeadDim := cfg.NValueDim
	kvLoraRank := 512
	kvBHeadDim := qkNope + vHeadDim

	if eng, ok := ds.Engine.(*Engine); ok {
		if v, ok2 := eng.Model.MetaU32("deepseek2.attention.kv_lora_rank"); ok2 {
			kvLoraRank = int(v)
		}
	}

	// 1) RMSNorm input
	normW := tensorF32Unsafe(layer.AttnNorm)
	copy(ds.AttnNormed[:cfg.NEmbd], ds.AttnCur[:cfg.NEmbd])
	rmsNorm(ds.AttnNormed[:cfg.NEmbd], normW[:cfg.NEmbd])

	// 2) Q projection
	qFull := make([]float32, nHead*qHeadDim)
	matvecAuto(qFull, layer.AttnQ, ds.AttnNormed[:cfg.NEmbd], cfg.NEmbd, nHead*qHeadDim)

	// 3) KV_a projection (compressed KV + rope tail)
	kvADim := kvLoraRank + qkRope
	kvA := make([]float32, kvADim)
	matvecAuto(kvA, layer.AttnKVA_MQA, ds.AttnNormed[:cfg.NEmbd], cfg.NEmbd, kvADim)

	if len(layer.AttnKVANorm) > 0 {
		kvNormW := tensorF32Unsafe(layer.AttnKVANorm)
		rmsNorm(kvA[:kvLoraRank], kvNormW[:kvLoraRank])
	}

	// 4) RoPE
	freqBase := cfg.RoPEFreqBase
	freqScale := float32(1.0)
	if cfg.RoPEScaleFactor > 0 {
		freqScale = 1.0 / cfg.RoPEScaleFactor
	}
	ropeYaRNTailInplace(qFull, pos, nHead, qHeadDim, qkRope, freqBase, freqScale, false)
	ropeYaRNTailInplace(kvA, pos, 1, kvADim, qkRope, freqBase, freqScale, false)

	// 5) Cache compressed KV row
	cache.PushRawKV(kvA)

	nRaw := cache.NRaw
	if nRaw > cache.CapRaw {
		nRaw = cache.CapRaw
	}
	if nRaw == 0 {
		for i := 0; i < cfg.NEmbd; i++ {
			ds.AttnOut[i] = 0
		}
		return
	}

	// 6) Expand cached lora rows through KV_b
	cachedExpanded := make([][]float32, nRaw)
	for t := 0; t < nRaw; t++ {
		row := cache.RawRow(t)
		exp := make([]float32, nHead*kvBHeadDim)
		matvecAuto(exp, layer.AttnKVB, row[:kvLoraRank], kvLoraRank, nHead*kvBHeadDim)
		cachedExpanded[t] = exp
	}

	// 7) Attention per head
	scale := float32(1.0 / math.Sqrt(float64(qHeadDim)))
	headsOut := make([]float32, nHead*vHeadDim)

	for h := 0; h < nHead; h++ {
		qNope := qFull[h*qHeadDim : h*qHeadDim+qkNope]
		qRope := qFull[h*qHeadDim+qkNope : (h+1)*qHeadDim]

		scores := make([]float32, nRaw)
		maxScore := float32(-1e30)

		for t := 0; t < nRaw; t++ {
			row := cache.RawRow(t)
			kNope := cachedExpanded[t][h*kvBHeadDim : h*kvBHeadDim+qkNope]
			kRope := row[kvLoraRank : kvLoraRank+qkRope]

			dot := float32(0)
			for i := 0; i < qkNope; i++ {
				dot += qNope[i] * kNope[i]
			}
			for i := 0; i < qkRope; i++ {
				dot += qRope[i] * kRope[i]
			}
			s := dot * scale
			scores[t] = s
			if s > maxScore {
				maxScore = s
			}
		}

		denom := float32(0)
		for i := range scores {
			e := float32(math.Exp(float64(scores[i] - maxScore)))
			scores[i] = e
			denom += e
		}
		if denom > 0 {
			inv := 1.0 / denom
			for i := range scores {
				scores[i] *= inv
			}
		}

		hOut := headsOut[h*vHeadDim : (h+1)*vHeadDim]
		for t := 0; t < nRaw; t++ {
			vHead := cachedExpanded[t][h*kvBHeadDim+qkNope : h*kvBHeadDim+qkNope+vHeadDim]
			w := scores[t]
			for i := 0; i < vHeadDim; i++ {
				hOut[i] += w * vHead[i]
			}
		}
	}

	// 8) Output projection
	matvecAuto(ds.AttnOut[:cfg.NEmbd], layer.AttnOutput, headsOut, nHead*vHeadDim, cfg.NEmbd)
}
