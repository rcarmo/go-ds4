//go:build darwin && cgo && metal

#include <stdbool.h>
#include <stdio.h>

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

#import "../../ds4_metal.m"
