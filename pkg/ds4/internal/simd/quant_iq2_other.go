//go:build !amd64

package simd

import "unsafe"

// DotIQ2Group32 scalar fallback.
func DotIQ2Group32(grid0, grid1, grid2, grid3 unsafe.Pointer, q8 unsafe.Pointer) int32 {
	g0 := (*[8]int8)(grid0)
	g1 := (*[8]int8)(grid1)
	g2 := (*[8]int8)(grid2)
	g3 := (*[8]int8)(grid3)
	q := (*[32]int8)(q8)
	sum := int32(0)
	for k := 0; k < 8; k++ {
		sum += int32(g0[k]) * int32(q[k])
		sum += int32(g1[k]) * int32(q[8+k])
		sum += int32(g2[k]) * int32(q[16+k])
		sum += int32(g3[k]) * int32(q[24+k])
	}
	return sum
}
