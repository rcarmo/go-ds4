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
	downIsQ2K    bool
	downIsQ5K    bool
	downIsIQ4NL  bool
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

	// Try Q2_K first (more common): rowBytes = ceil(nEmbd/256)*84
	q2kRowBytes := ((cfg.NEmbd + 255) / 256) * BlockQ2KSize
	if q2kRowBytes > 0 {
		nFFExpQ2K := gatePerExpert / q2kRowBytes
		if nFFExpQ2K*q2kRowBytes == gatePerExpert && nFFExpQ2K > 0 {
			ed.outDim = nFFExpQ2K
			ed.gateRowBytes = q2kRowBytes
			ed.upRowBytes = q2kRowBytes
			ed.gateIsIQ2 = false
		}
	}

	// Try IQ2_XXS: rowBytes = ceil(nEmbd/256)*66
	if ed.outDim == 0 {
		iq2RowBytes := ((cfg.NEmbd + 255) / 256) * BlockIQ2XXSSize
		if iq2RowBytes > 0 {
			nFFExpIQ2 := gatePerExpert / iq2RowBytes
			if nFFExpIQ2*iq2RowBytes == gatePerExpert && nFFExpIQ2 > 0 {
				ed.outDim = nFFExpIQ2
				ed.gateRowBytes = iq2RowBytes
				ed.upRowBytes = iq2RowBytes
				ed.gateIsIQ2 = true
			}
		}
	}

	// Detect down tensor format
	if len(layer.FfnDownExps) > 0 && ed.outDim > 0 {
		downPerExpert := len(layer.FfnDownExps) / cfg.NExpert
		// Try Q2_K: (nFFExp/256)*84 bytes per row, nEmbd rows
		q2kDown := ((ed.outDim + 255) / 256) * BlockQ2KSize
		if q2kDown > 0 && cfg.NEmbd*q2kDown == downPerExpert {
			ed.downRowBytes = q2kDown
			ed.downIsQ2K = true
		}
		// Try Q5_K: ceil(nFFExp/256)*176 bytes per row, nEmbd rows
		if ed.downRowBytes == 0 {
			q5kDown := ((ed.outDim + 255) / 256) * BlockQ5KSize
			if q5kDown > 0 && cfg.NEmbd*q5kDown == downPerExpert {
				ed.downRowBytes = q5kDown
				ed.downIsQ5K = true
			}
		}
		// Try IQ4_NL: ceil(nFFExp/32)*18 bytes per row
		if ed.downRowBytes == 0 {
			iq4Down := ((ed.outDim + QK4_NL - 1) / QK4_NL) * BlockIQ4NLSize
			if iq4Down > 0 && cfg.NEmbd*iq4Down == downPerExpert {
				ed.downRowBytes = iq4Down
				ed.downIsQ5K = false // reuse field: not Q5K, it's IQ4NL
				ed.downIsIQ4NL = true
			}
		}
		if ed.downRowBytes == 0 {
			ed.downRowBytes = downPerExpert / cfg.NEmbd
		}
	}

	return ed
}
