// mmap_linux.go — Linux madvise support.
//
//go:build linux

package ds4

import "syscall"

func madviseRaw(b []byte, advice int) error {
	return syscall.Madvise(b, advice)
}
