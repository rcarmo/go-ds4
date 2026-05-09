package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/rcarmo/go-ds4/pkg/ds4"
)

func main() {
	modelPath := flag.String("model", "gguf/ds4-q2.gguf", "GGUF model path")
	ctxSize := flag.Int("ctx", 4096, "context size")
	maxTokens := flag.Int("n", 256, "max tokens per response")
	temp := flag.Float64("temp", 0.7, "temperature (0=greedy)")
	topK := flag.Int("topk", 40, "top-k sampling")
	fast := flag.Bool("fast", false, "fast expert mode")
	useGPU := flag.Bool("gpu", false, "enable GPU")
	strictGPU := flag.Bool("gpu-strict", false, "require GPU kernels; panic/error instead of CPU fallback for GPU-covered paths")
	flag.Parse()

	fmt.Printf("Loading %s...\n", *modelPath)
	t0 := time.Now()
	engine, err := ds4.OpenEngineWithOptions(ds4.EngineOptions{
		ModelPath:   *modelPath,
		FastExperts: *fast,
		UseGPU:      *useGPU || *strictGPU,
		StrictGPU:   *strictGPU,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()
	fmt.Printf("Ready in %v (%s)\n", time.Since(t0), engine.Config)
	fmt.Printf("Vocab: %d tokens\n\n", len(engine.Vocab.Tokens))

	session := engine.NewSession(*ctxSize)
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "/quit" || input == "/exit" {
			break
		}
		if input == "/reset" {
			session.Invalidate()
			session = engine.NewSession(*ctxSize)
			fmt.Println("[session reset]")
			continue
		}

		tokens := engine.Vocab.EncodeChatPrompt("", input, false)

		// Prefill
		t1 := time.Now()
		session.Prefill(tokens)
		prefillDur := time.Since(t1)

		// Decode
		t2 := time.Now()
		generated := 0
		var out strings.Builder
		for i := 0; i < *maxTokens; i++ {
			var tok int
			if *temp <= 0 {
				tok = ds4.Argmax(session.Logits)
			} else {
				tok = ds4.Sample(session.Logits, float32(*temp), *topK, 0.9, 0)
			}
			if tok == engine.Vocab.EOS {
				break
			}
			text := engine.Vocab.TokenText(tok)
			out.WriteString(text)
			fmt.Print(text)
			session.Eval(tok)
			generated++
		}
		decodeDur := time.Since(t2)
		fmt.Println()

		prefillTokS := float64(len(tokens)) / prefillDur.Seconds()
		decodeTokS := float64(generated) / decodeDur.Seconds()
		fmt.Printf("[prefill: %d tok, %.1f tok/s | decode: %d tok, %.1f tok/s | %.0fms total]\n\n",
			len(tokens), prefillTokS, generated, decodeTokS,
			float64((prefillDur + decodeDur).Milliseconds()))
	}
}
