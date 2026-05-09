package ds4

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// Vocab holds the tokenizer state loaded from GGUF metadata.
type Vocab struct {
	Tokens   []string       // token ID → string
	TokenMap map[string]int // string → token ID
	Merges   map[string]int // "tok1 tok2" → merge rank

	BOS        int // beginning of sequence
	EOS        int // end of sequence
	User       int // <|User|>
	Assistant  int // <|Assistant|>
	ThinkStart int // <think>
	ThinkEnd   int // </think>
}

// LoadVocab builds the tokenizer from GGUF metadata.
func LoadVocab(m *GGUFModel) (*Vocab, error) {
	tokens, ok := m.MetaStrArray("tokenizer.ggml.tokens")
	if !ok {
		return nil, errorf("missing tokenizer.ggml.tokens")
	}

	v := &Vocab{
		Tokens:   tokens,
		TokenMap: make(map[string]int, len(tokens)),
		Merges:   make(map[string]int),
	}

	for i, t := range tokens {
		v.TokenMap[t] = i
	}

	// Merge ranks
	merges, ok := m.MetaStrArray("tokenizer.ggml.merges")
	if ok {
		for i, merge := range merges {
			v.Merges[merge] = i
		}
	}

	// Special token IDs
	v.BOS = v.findToken("<｜begin▁of▁sentence｜>", 0)
	v.EOS = v.findToken("<｜end▁of▁sentence｜>", 1)
	v.User = v.findToken("<｜User｜>", -1)
	v.Assistant = v.findToken("<｜Assistant｜>", -1)
	v.ThinkStart = v.findToken("<think>", -1)
	v.ThinkEnd = v.findToken("</think>", -1)

	return v, nil
}

func (v *Vocab) findToken(s string, fallback int) int {
	if id, ok := v.TokenMap[s]; ok {
		return id
	}
	return fallback
}

// gpt2ByteToCodepoint maps a raw byte to its GPT-2 Unicode codepoint.
// Printable ASCII and most Latin-1 characters map to themselves.
// The remaining 100 "control" bytes map to codepoints 256-355.
func gpt2ByteToCodepoint(b byte) rune {
	if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 {
		return rune(b)
	}
	// Count how many "self-mapping" bytes come before b
	n := 0
	for x := 0; x < 256; x++ {
		xb := byte(x)
		if (xb >= 33 && xb <= 126) || (xb >= 161 && xb <= 172) || xb >= 174 {
			continue
		}
		if xb == b {
			return rune(256 + n)
		}
		n++
	}
	return rune(b) // unreachable
}

// init precomputes the byte→codepoint table for speed.
var gpt2ByteTable [256]rune

func init() {
	for i := 0; i < 256; i++ {
		gpt2ByteTable[i] = gpt2ByteToCodepoint(byte(i))
	}
}

// byteEncode maps raw bytes to GPT-2 Unicode codepoints (UTF-8 string).
func byteEncode(data []byte) string {
	var buf strings.Builder
	buf.Grow(len(data) * 2)
	for _, b := range data {
		buf.WriteRune(gpt2ByteTable[b])
	}
	return buf.String()
}

// bpeRank looks up the merge rank for two adjacent symbols.
func (v *Vocab) bpeRank(a, b string) (int, bool) {
	key := a + " " + b
	rank, ok := v.Merges[key]
	return rank, ok
}

// bpeMerge applies BPE to a single pre-tokenized piece and appends token IDs to out.
func (v *Vocab) bpeMerge(piece string, out *[]int) {
	if len(piece) == 0 {
		return
	}

	// Byte-encode the piece
	encoded := byteEncode([]byte(piece))

	// Split into individual UTF-8 characters as initial symbols
	symbols := make([]string, 0, utf8.RuneCountInString(encoded))
	for _, r := range encoded {
		symbols = append(symbols, string(r))
	}

	if len(symbols) == 1 {
		if id, ok := v.TokenMap[symbols[0]]; ok {
			*out = append(*out, id)
		}
		return
	}

	// Iteratively merge the lowest-rank pair
	for len(symbols) > 1 {
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(symbols)-1; i++ {
			if rank, ok := v.bpeRank(symbols[i], symbols[i+1]); ok {
				if bestRank < 0 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx < 0 {
			break // no more merges
		}
		// Merge symbols[bestIdx] and symbols[bestIdx+1]
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Look up each surviving symbol
	for _, sym := range symbols {
		if id, ok := v.TokenMap[sym]; ok {
			*out = append(*out, id)
		}
		// Unknown symbols are silently dropped (matching ds4.c behavior)
	}
}

// isAlpha returns true for ASCII letters + CJK ideographs + kana.
func isAlpha(r rune) bool {
	if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
		return true
	}
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) || unicode.Is(unicode.Katakana, r)
}

// Tokenize converts text to token IDs using GPT-2 BPE with JoyAI pre-tokenization.
func (v *Vocab) Tokenize(text string) []int {
	out := make([]int, 0, len(text)/4)

	// Check for special tokens first
	for len(text) > 0 {
		// Try to match a special token at current position
		specialLen := v.matchSpecialToken(text)
		if specialLen > 0 {
			token := text[:specialLen]
			if id, ok := v.TokenMap[token]; ok {
				out = append(out, id)
			}
			text = text[specialLen:]
			continue
		}

		// Find next special token or end of text
		nextSpecial := len(text)
		for i := 1; i < len(text); i++ {
			if v.matchSpecialToken(text[i:]) > 0 {
				nextSpecial = i
				break
			}
		}

		// Pre-tokenize the non-special segment
		segment := text[:nextSpecial]
		text = text[nextSpecial:]
		v.joyAIPreTokenize(segment, &out)
	}

	return out
}

// matchSpecialToken checks if text starts with a known special token.
func (v *Vocab) matchSpecialToken(text string) int {
	specials := []string{
		"<｜begin▁of▁sentence｜>",
		"<｜end▁of▁sentence｜>",
		"<｜User｜>",
		"<｜Assistant｜>",
		"<think>",
		"</think>",
	}
	for _, s := range specials {
		if strings.HasPrefix(text, s) {
			return len(s)
		}
	}
	return 0
}

// joyAIPreTokenize splits text into pieces using JoyAI/DeepSeek rules.
// Each piece is then BPE-merged independently.
func (v *Vocab) joyAIPreTokenize(text string, out *[]int) {
	runes := []rune(text)
	n := len(runes)
	i := 0

	for i < n {
		r := runes[i]

		// Apostrophe contractions: 's, 't, 're, 've, 'm, 'll, 'd
		if r == '\'' && i+1 < n {
			suffixes := []string{"s", "t", "re", "ve", "m", "ll", "d"}
			matched := false
			for _, sfx := range suffixes {
				end := i + 1 + len(sfx)
				if end <= n && string(runes[i+1:end]) == sfx {
					piece := string(runes[i:end])
					v.bpeMerge(piece, out)
					i = end
					matched = true
					break
				}
			}
			if matched {
				continue
			}
		}

		// Optional leading space + alphabetic run
		if r == ' ' && i+1 < n && isAlpha(runes[i+1]) {
			j := i + 1
			for j < n && isAlpha(runes[j]) {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}
		if isAlpha(r) {
			j := i + 1
			for j < n && isAlpha(runes[j]) {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}

		// Digit runs (optionally preceded by space)
		if r == ' ' && i+1 < n && runes[i+1] >= '0' && runes[i+1] <= '9' {
			j := i + 1
			for j < n && runes[j] >= '0' && runes[j] <= '9' {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}
		if r >= '0' && r <= '9' {
			j := i + 1
			for j < n && runes[j] >= '0' && runes[j] <= '9' {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}

		// Whitespace runs
		if r == ' ' || r == '\t' || r == '\r' {
			j := i + 1
			for j < n && (runes[j] == ' ' || runes[j] == '\t' || runes[j] == '\r') {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}

		// Newlines (merge trailing newlines into punctuation piece)
		if r == '\n' {
			j := i + 1
			for j < n && runes[j] == '\n' {
				j++
			}
			v.bpeMerge(string(runes[i:j]), out)
			i = j
			continue
		}

		// Single character (punctuation, symbols, etc.)
		v.bpeMerge(string(r), out)
		i++
	}
}

// EncodeChatPrompt builds a full chat prompt with special tokens.
func (v *Vocab) EncodeChatPrompt(system, userPrompt string, thinkMode bool) []int {
	tokens := []int{v.BOS}

	if v.User < 0 {
		// Text-based chat template (V2 Lite style)
		if system != "" {
			tokens = append(tokens, v.Tokenize(system+"\n\n")...)
		}
		if userPrompt != "" {
			tokens = append(tokens, v.Tokenize("User: "+userPrompt+"\n\nAssistant:")...)
		}
		return tokens
	}

	if system != "" {
		tokens = append(tokens, v.User)
		tokens = append(tokens, v.Tokenize(system)...)
	}

	if userPrompt != "" {
		if system == "" {
			tokens = append(tokens, v.User)
		}
		tokens = append(tokens, v.Tokenize(userPrompt)...)
	}

	tokens = append(tokens, v.Assistant)
	if thinkMode && v.ThinkStart >= 0 {
		tokens = append(tokens, v.ThinkStart)
	} else if !thinkMode && v.ThinkEnd >= 0 {
		tokens = append(tokens, v.ThinkEnd)
	}

	return tokens
}

// TokenText returns the text for a token ID.
func (v *Vocab) TokenText(id int) string {
	if id < 0 || id >= len(v.Tokens) {
		return ""
	}
	return DecodeTokenText(v.Tokens[id])
}

func errorf(format string, args ...interface{}) error {
	return &vocabError{msg: sprintf(format, args...)}
}

type vocabError struct{ msg string }

func (e *vocabError) Error() string { return e.msg }

func sprintf(format string, args ...interface{}) string {
	if len(args) == 0 {
		return format
	}
	return format // simplified — no fmt dependency
}

// gpt2DecodeTable maps GPT-2 Unicode codepoints back to raw bytes.
var gpt2DecodeTable map[rune]byte

func init() {
	gpt2DecodeTable = make(map[rune]byte, 256)
	for i := 0; i < 256; i++ {
		gpt2DecodeTable[gpt2ByteTable[i]] = byte(i)
	}
}

// DecodeTokenText converts a GPT-2 BPE token string back to UTF-8.
func DecodeTokenText(s string) string {
	raw := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := gpt2DecodeTable[r]; ok {
			raw = append(raw, b)
		} else {
			// Pass through non-BPE runes as UTF-8
			buf := make([]byte, 4)
			n := copy(buf, []byte(string(r)))
			raw = append(raw, buf[:n]...)
		}
	}
	return string(raw)
}
