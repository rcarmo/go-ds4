package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/rcarmo/go-ds4/pkg/ds4"
)

// OpenAI API types

type ChatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	Temperature float32       `json:"temperature"`
	TopK        int           `json:"top_k"`
	TopP        float32       `json:"top_p"`
	MaxTokens   int           `json:"max_tokens"`
}

type ChatCompletionChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   Usage                  `json:"usage"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// SSE streaming types

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason"`
}

type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

// Model list

type ModelObject struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelListResponse struct {
	Object string        `json:"object"`
	Data   []ModelObject `json:"data"`
}

// Server state

type Server struct {
	engine  *ds4.Engine
	session *ds4.Session
	mu      sync.Mutex
	model   string
}

func newServer(engine *ds4.Engine, ctxSize int, model string) *Server {
	return &Server{
		engine:  engine,
		session: engine.NewSession(ctxSize),
		model:   model,
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	resp := ModelListResponse{
		Object: "list",
		Data: []ModelObject{
			{
				ID:      s.model,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "local",
			},
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %v", err), http.StatusBadRequest)
		return
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}
	temp := req.Temperature
	topK := req.TopK
	if topK <= 0 {
		topK = 40
	}

	// Build prompt from messages
	var system, userMsg string
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			system += msg.Content
		case "user":
			userMsg = msg.Content
		}
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Invalidate session for fresh generation
	s.session.Invalidate()

	// Tokenize
	tokens := s.engine.Vocab.EncodeChatPrompt(system, userMsg, false)

	s.session.Prefill(tokens)
	promptTokens := len(tokens)

	if req.Stream {
		s.streamResponse(w, r, &req, promptTokens, maxTokens, temp, topK)
	} else {
		s.nonStreamResponse(w, &req, promptTokens, maxTokens, temp, topK)
	}
}

func (s *Server) nonStreamResponse(w http.ResponseWriter, req *ChatCompletionRequest, promptTokens, maxTokens int, temp float32, topK int) {
	var out strings.Builder
	generated := 0
	finishReason := "stop"

	for i := 0; i < maxTokens; i++ {
		token := ds4.Sample(s.session.Logits, temp, topK, req.TopP, 0)
		if token == s.engine.Vocab.EOS {
			break
		}
		out.WriteString(s.engine.Vocab.TokenText(token))
		s.session.Eval(token)
		generated++
	}
	if generated >= maxTokens {
		finishReason = "length"
	}

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	resp := ChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.model,
		Choices: []ChatCompletionChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: out.String(),
				},
				FinishReason: finishReason,
			},
		},
		Usage: Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: generated,
			TotalTokens:      promptTokens + generated,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) streamResponse(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest, promptTokens, maxTokens int, temp float32, topK int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	// Initial chunk with role
	writeSSE(w, flusher, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   s.model,
		Choices: []StreamChoice{
			{Index: 0, Delta: StreamDelta{Role: "assistant"}},
		},
	})

	generated := 0
	finishReason := "stop"

	for i := 0; i < maxTokens; i++ {
		// Check client disconnect
		select {
		case <-r.Context().Done():
			return
		default:
		}

		token := ds4.Sample(s.session.Logits, temp, topK, req.TopP, 0)
		if token == s.engine.Vocab.EOS {
			break
		}

		text := s.engine.Vocab.TokenText(token)
		writeSSE(w, flusher, StreamChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   s.model,
			Choices: []StreamChoice{
				{Index: 0, Delta: StreamDelta{Content: text}},
			},
		})

		s.session.Eval(token)
		generated++
	}
	if generated >= maxTokens {
		finishReason = "length"
	}

	// Final chunk with finish_reason
	writeSSE(w, flusher, StreamChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   s.model,
		Choices: []StreamChoice{
			{Index: 0, Delta: StreamDelta{}, FinishReason: &finishReason},
		},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeSSE(w io.Writer, flusher http.Flusher, chunk StreamChunk) {
	data, _ := json.Marshal(chunk)
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func main() {
	modelPath := flag.String("model", "gguf/ds4-q2.gguf", "path to GGUF model file")
	listen := flag.String("listen", ":8080", "address to listen on")
	ctxSize := flag.Int("ctx", 4096, "context size")
	fast := flag.Bool("fast", false, "use top-4 experts (faster, slight quality loss)")
	useGPU := flag.Bool("gpu", false, "enable GPU")
	strictGPU := flag.Bool("gpu-strict", false, "require GPU kernels; no CPU fallback for GPU-covered paths")
	flag.Parse()

	log.Printf("Loading model from %s...", *modelPath)
	engine, err := ds4.OpenEngineWithOptions(ds4.EngineOptions{
		ModelPath:   *modelPath,
		FastExperts: *fast,
		UseGPU:      *useGPU || *strictGPU,
		StrictGPU:   *strictGPU,
	})
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer engine.Close()

	modelName := "deepseek-v4-flash"
	server := newServer(engine, *ctxSize, modelName)

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", server.handleModels)
	mux.HandleFunc("/v1/models/"+modelName, server.handleModels)
	mux.HandleFunc("/v1/chat/completions", server.handleChatCompletions)

	log.Printf("Server listening on %s", *listen)
	log.Printf("  POST /v1/chat/completions (OpenAI compatible)")
	log.Printf("  GET  /v1/models")
	if *fast {
		log.Printf("  Mode: fast (top-4 experts)")
	}

	if err := http.ListenAndServe(*listen, mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
