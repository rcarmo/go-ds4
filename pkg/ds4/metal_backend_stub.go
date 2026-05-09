//go:build !darwin || !cgo || !metal

package ds4

import "fmt"

func (e *Engine) initMetalGPU() (interface{}, error) {
	return nil, fmt.Errorf("Metal backend not built; rebuild on macOS with CGO_ENABLED=1 -tags metal")
}

func metalGPUClose(g interface{}) bool {
	return false
}

func metalGPUReady(g interface{}) (bool, bool) {
	return false, false
}

func metalGPUMatvecQ8_0(g interface{}, out []float32, tensorName string, x []float32, inDim, outDim int) (bool, bool) {
	return false, false
}

func metalGPUMatvecQ8_0Grouped(g interface{}, out []float32, tensorName string, x []float32, inDim, outDim, groupSize int) (bool, bool) {
	return false, false
}

func metalGPUExpertForward(g interface{}, ds *DecodeState, layer *LayerWeights, experts []expertScore, il int) ([]bool, bool) {
	return nil, false
}

func metalGPUPrefill(g interface{}, s *Session, tokens []int) (bool, bool) {
	return false, false
}

func metalGPUSerializes(g interface{}) bool {
	return false
}
