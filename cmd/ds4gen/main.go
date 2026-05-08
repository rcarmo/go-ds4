package main

import (
	"fmt"
	"os"
	"time"

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

	fmt.Printf("Opening %s...\n", path)
	t0 := time.Now()
	m, err := ds4.OpenGGUF(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "GGUF open error: %v\n", err)
		os.Exit(1)
	}
	defer m.Close()
	fmt.Printf("GGUF parsed in %v: %d tensors, %d metadata keys\n",
		time.Since(t0), len(m.Tensors), len(m.Meta))

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
		"deepseek4.hyper_connection.count",
		"tokenizer.ggml.model",
	} {
		if v, ok := m.Meta[key]; ok {
			fmt.Printf("  %s = %v\n", key, v)
		}
	}

	// Show tensor summary
	fmt.Printf("\nTensor types:\n")
	typeCounts := make(map[uint32]int)
	var totalBytes uint64
	for _, t := range m.Tensors {
		typeCounts[t.Type]++
		totalBytes += t.DataBytes()
	}
	typeNames := map[uint32]string{0: "F32", 1: "F16", 8: "Q8_0", 10: "Q2_K", 12: "Q4_K", 16: "IQ2_XXS", 26: "I32"}
	for typ, count := range typeCounts {
		name := typeNames[typ]
		if name == "" {
			name = fmt.Sprintf("type_%d", typ)
		}
		fmt.Printf("  %s: %d tensors\n", name, count)
	}
	fmt.Printf("  Total tensor data: %.1f GB\n", float64(totalBytes)/1024/1024/1024)

	// Memory estimate
	mem := ds4.EstimateMemory(4096)
	fmt.Printf("\nMemory estimate (ctx=4096):\n")
	fmt.Printf("  Non-expert: %.1f GB\n", mem.NonExpertMB/1024)
	fmt.Printf("  Expert: %.1f GB\n", mem.ExpertMB/1024)
	fmt.Printf("  Active set: %.1f GB\n", mem.ActiveSetMB/1024)
	fmt.Printf("  KV cache: %.1f MB\n", mem.KVCacheMB)

	// Try full engine load with streaming
	fmt.Println("\nOpening engine (streaming mode)...")
	t0 = time.Now()
	engine, err := ds4.OpenEngineWithOptions(ds4.EngineOptions{
		ModelPath:     path,
		StreamExperts: true,
	})
	if err != nil {
		fmt.Printf("  Engine open failed: %v\n", err)
		fmt.Println("\n(This is expected for partial downloads)")
		return
	}
	defer engine.Close()
	fmt.Printf("Engine ready in %v\n", time.Since(t0))
	fmt.Printf("  Vocab: %d tokens, BOS=%d EOS=%d\n",
		len(engine.Vocab.Tokens), engine.Vocab.BOS, engine.Vocab.EOS)

	// Tokenize
	tokens := engine.Vocab.EncodeChatPrompt("", prompt, false)
	fmt.Printf("\nPrompt: %q → %d tokens\n", prompt, len(tokens))
	for i, t := range tokens {
		if i > 10 {
			fmt.Printf("  ... (%d more)\n", len(tokens)-10)
			break
		}
		fmt.Printf("  [%d] %d = %q\n", i, t, engine.Vocab.TokenText(t))
	}

	// Create session and try eval
	sess := engine.NewSession(4096)
	fmt.Printf("\nSession: ctx=%d\n", sess.CtxSize)

	fmt.Println("\nEval first token...")
	t0 = time.Now()
	sess.Eval(tokens[0])
	fmt.Printf("First token eval: %v\n", time.Since(t0))

	// Argmax from logits
	topToken := ds4.Argmax(sess.Logits)
	fmt.Printf("Top logit: token %d = %q\n", topToken, engine.Vocab.TokenText(topToken))
}
