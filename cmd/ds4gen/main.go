package main

import (
	"fmt"
	"os"

	"github.com/rcarmo/go-ds4/ds4"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: %s <model.gguf>\n", os.Args[0])
		os.Exit(1)
	}
	path := os.Args[1]

	fmt.Printf("Opening %s...\n", path)
	m, err := ds4.OpenGGUF(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()

	fmt.Printf("GGUF loaded: %d tensors, %d metadata keys\n", len(m.Tensors), len(m.Meta))

	// Print key metadata
	for _, key := range []string{
		"general.architecture",
		"general.name",
		"deepseek4.block_count",
		"deepseek4.embedding_length",
		"deepseek4.attention.head_count",
		"deepseek4.attention.head_count_kv",
		"deepseek4.expert_count",
		"deepseek4.expert_used_count",
	} {
		if v, ok := m.Meta[key]; ok {
			fmt.Printf("  %s = %v\n", key, v)
		}
	}

	// Try binding weights
	fmt.Println("\nBinding weights...")
	w, err := ds4.BindWeights(m)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bind error: %v\n", err)
		os.Exit(1)
	}

	// Print tensor stats
	fmt.Printf("  token_embd: %d bytes\n", len(w.TokenEmbd))
	fmt.Printf("  output: %d bytes\n", len(w.Output))
	fmt.Printf("  layer[0].attn_q_a: %d bytes\n", len(w.Layer[0].AttnQA))
	fmt.Printf("  layer[0].ffn_gate_exps: %d bytes\n", len(w.Layer[0].FfnGateExps))

	// Count optional tensors
	nCompressor := 0
	nIndexer := 0
	nHash := 0
	for il := 0; il < ds4.NLayer; il++ {
		if w.Layer[il].CompressorKV != nil {
			nCompressor++
		}
		if w.Layer[il].IndexerQB != nil {
			nIndexer++
		}
		if w.Layer[il].FfnGateTid2Eid != nil {
			nHash++
		}
	}
	fmt.Printf("\n  Compressor layers: %d\n", nCompressor)
	fmt.Printf("  Indexer layers: %d\n", nIndexer)
	fmt.Printf("  Hash routing layers: %d\n", nHash)

	fmt.Println("\nDone.")
}
