package ds4

// expertDims detects expert dimensions and quant format from tensor data sizes.
type expertDims struct {
	inDim        int // expert input dimension (NEmbd for gate/up, NFFExp for down)
	outDim       int // expert output dimension (NFFExp for gate/up, NEmbd for down)
	nExperts     int // number of experts
	gateRowBytes int // bytes per gate output row
	upRowBytes   int
	downRowBytes int
	gateIsIQ2    bool // true if IQ2_XXS, false if Q2_K
	downIsQ2K    bool // true if Q2_K, false if other
}

func detectExpertDims(layer *LayerWeights, cfg *ModelConfig) expertDims {
	ed := expertDims{
		inDim:    cfg.NEmbd,
		nExperts: cfg.NExpert,
	}

	if len(layer.FfnGateExps) == 0 {
		return ed
	}

	// Detect FFN dim and format from gate tensor
	// gate: [nFFExp, nExpert * nEmbd] — 3D tensor flattened
	// Per expert: gatePerExpert bytes
	gatePerExpert := len(layer.FfnGateExps) / cfg.NExpert

	// Try IQ2_XXS: rowBytes = (nEmbd/256)*66
	iq2RowBytes := (cfg.NEmbd / 256) * BlockIQ2XXSSize
	if iq2RowBytes > 0 {
		nFFExpIQ2 := gatePerExpert / iq2RowBytes
		if nFFExpIQ2*iq2RowBytes == gatePerExpert && nFFExpIQ2 > 0 {
			ed.outDim = nFFExpIQ2
			ed.gateRowBytes = iq2RowBytes
			ed.upRowBytes = iq2RowBytes
			ed.gateIsIQ2 = true
		}
	}

	// Try Q2_K: rowBytes = (nEmbd/256)*84
	if ed.outDim == 0 {
		q2kRowBytes := (cfg.NEmbd / 256) * BlockQ2KSize
		if q2kRowBytes > 0 {
			nFFExpQ2K := gatePerExpert / q2kRowBytes
			if nFFExpQ2K*q2kRowBytes == gatePerExpert && nFFExpQ2K > 0 {
				ed.outDim = nFFExpQ2K
				ed.gateRowBytes = q2kRowBytes
				ed.upRowBytes = q2kRowBytes
				ed.gateIsIQ2 = false
			}
		}
	}

	// Detect down tensor format
	if len(layer.FfnDownExps) > 0 {
		downPerExpert := len(layer.FfnDownExps) / cfg.NExpert
		// down: [nEmbd, nFFExp] — output is nEmbd, input is nFFExp
		// Try Q2_K with nFFExp as input
		q2kDown := (ed.outDim / 256) * BlockQ2KSize
		if q2kDown > 0 && cfg.NEmbd*q2kDown == downPerExpert {
			ed.downRowBytes = q2kDown
			ed.downIsQ2K = true
		}
		// Try other formats (Q5_K_M = type 20, not yet supported — use raw bytes)
		if ed.downRowBytes == 0 {
			// Fallback: compute from total size
			ed.downRowBytes = downPerExpert / cfg.NEmbd
		}
	}

	return ed
}
