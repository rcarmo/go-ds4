# API And Server Usage

`go-ds4` can be used as a Go package, an interactive chat CLI, or an OpenAI-compatible HTTP server.

## Go API

```go
package main

import (
	"fmt"
	"log"

	"github.com/rcarmo/go-ds4/pkg/ds4"
)

func main() {
	engine, err := ds4.OpenEngineWithOptions(ds4.EngineOptions{
		ModelPath:   "model.gguf",
		FastExperts: true,
		UseGPU:      true,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	session := engine.NewSession(4096)
	tokens := engine.Vocab.EncodeChatPrompt("", "Why is the sky blue?", false)
	session.Prefill(tokens)

	for i := 0; i < 100; i++ {
		tok := ds4.Sample(session.Logits, 0.7, 40, 0.9, 0)
		if tok == engine.Vocab.EOS {
			break
		}
		fmt.Print(engine.Vocab.TokenText(tok))
		session.Eval(tok)
	}
}
```

## Interactive CLI

```bash
go build ./cmd/ds4chat
./ds4chat -model /path/to/model.gguf
```

Common options:

```bash
./ds4chat -model model.gguf -ctx 8192 -temp 0.7 -topk 40
./ds4chat -model model.gguf -gpu
./ds4chat -model model.gguf -fast
```

## OpenAI-Compatible Server

Start the server:

```bash
go run ./cmd/ds4server -model /path/to/model.gguf -listen :8080
```

Call the chat completions endpoint:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

The server is intentionally small and follows the package API. It is useful for local experiments and compatibility testing with clients that expect OpenAI-style endpoints.
