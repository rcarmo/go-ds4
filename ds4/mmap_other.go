// mmap_other.go — Stub for non-Linux platforms.
//
//go:build !linux

package ds4

func madviseRaw(b []byte, advice int) error {
	return nil // madvise not available, silently ignore
}
