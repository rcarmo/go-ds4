package main

import (
	"fmt"
	"os"

	"github.com/rcarmo/go-ds4/ds4"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model.gguf> [prompt]\n", os.Args[0])
		os.Exit(1)
	}
	path := os.Args[1]
	prompt := "Hello"
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	fmt.Printf("Loading %s...\n", path)
	engine, err := ds4.OpenEngine(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()

	fmt.Printf("Model: %d tensors, vocab: %d tokens\n",
		len(engine.Model.Tensors), len(engine.Vocab.Tokens))
	fmt.Printf("BOS=%d EOS=%d\n", engine.Vocab.BOS, engine.Vocab.EOS)

	// Count optional features
	nComp, nIdx, nHash := 0, 0, 0
	for il := 0; il < ds4.NLayer; il++ {
		if engine.Weights.Layer[il].CompressorKV != nil {
			nComp++
		}
		if engine.Weights.Layer[il].IndexerQB != nil {
			nIdx++
		}
		if engine.Weights.Layer[il].FfnGateTid2Eid != nil {
			nHash++
		}
	}
	fmt.Printf("Compressor layers: %d, Indexer: %d, Hash: %d\n", nComp, nIdx, nHash)

	// Tokenize
	tokens := engine.Vocab.EncodeChatPrompt("", prompt, false)
	fmt.Printf("\nPrompt: %q → %d tokens\n", prompt, len(tokens))
	for i, t := range tokens {
		fmt.Printf("  [%d] %d = %q\n", i, t, engine.Vocab.TokenText(t))
	}

	// Create session
	sess := engine.NewSession(4096)
	fmt.Printf("\nSession: ctx=%d\n", sess.CtxSize)

	// Generate (if model is loaded — will crash without actual GGUF weights bound)
	fmt.Println("\nReady for generation (needs actual model weights).")
}
