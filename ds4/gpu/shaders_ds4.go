package gpu

// DS4 Q8_0 GEMV compute shader (GLSL source for documentation).
//
// To compile: glslangValidator -V ds4_gemv_q8_0.comp -o ds4_gemv_q8_0.spv
//
// DS4 Q8_0 block layout: f16 scale (2 bytes) + int8[32] (34 bytes total).
// One workgroup (256 threads) computes one output row.
// Each thread handles ceil(nBlocks/256) blocks, reducing via shared memory.

// GLSL source:
//
// #version 450
// #extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
// #extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
//
// layout(local_size_x = 256) in;
//
// layout(set=0, binding=0) readonly buffer Act { float activation[]; };
// layout(set=0, binding=1) readonly buffer Wt  { uint  weight_raw[]; }; // raw bytes
// layout(set=0, binding=2) writeonly buffer Out { float output[]; };
//
// layout(push_constant) uniform Params {
//     uint inDim;    // input dimension (e.g. 4096)
//     uint outDim;   // output dimension
//     uint rowBytes; // bytes per weight row (nBlocks * 34)
// };
//
// shared float sdata[256];
//
// void main() {
//     uint row = gl_WorkGroupID.x;
//     uint tid = gl_LocalInvocationID.x;
//     if (row >= outDim) return;
//
//     uint nBlocks = inDim / 32;
//     uint rowOff = row * rowBytes;  // byte offset for this row
//
//     float sum = 0.0;
//     for (uint b = tid; b < nBlocks; b += 256) {
//         // Block layout: 2 bytes f16 scale + 32 bytes int8
//         uint blockOff = rowOff + b * 34;
//
//         // Read f16 scale (2 bytes at blockOff)
//         uint scaleWord = weight_raw[blockOff / 4];
//         uint shift = (blockOff % 4) * 8;
//         uint scaleBits = (scaleWord >> shift) & 0xFFFF;
//         float scale = unpackHalf2x16(scaleBits).x;
//
//         // Dot int8[32] × float32[32]
//         float blockDot = 0.0;
//         uint qsOff = blockOff + 2;
//         uint actOff = b * 32;
//         for (uint i = 0; i < 32; i += 4) {
//             uint packed = weight_raw[(qsOff + i) / 4];
//             uint byteShift = ((qsOff + i) % 4) * 8;
//             // Extract 4 signed int8 values
//             int q0 = int(int8_t(bitfieldExtract(packed, int(byteShift), 8)));
//             int q1 = int(int8_t(bitfieldExtract(packed, int(byteShift+8), 8)));
//             int q2 = int(int8_t(bitfieldExtract(packed, int(byteShift+16), 8)));
//             int q3 = int(int8_t(bitfieldExtract(packed, int(byteShift+24), 8)));
//             blockDot += float(q0) * activation[actOff + i]
//                       + float(q1) * activation[actOff + i + 1]
//                       + float(q2) * activation[actOff + i + 2]
//                       + float(q3) * activation[actOff + i + 3];
//         }
//         sum += scale * blockDot;
//     }
//
//     sdata[tid] = sum;
//     barrier();
//
//     // Tree reduction
//     for (uint s = 128; s > 0; s >>= 1) {
//         if (tid < s) sdata[tid] += sdata[tid + s];
//         barrier();
//     }
//
//     if (tid == 0) output[row] = sdata[0];
// }
//
// Note: the above uses byte-level access which requires careful alignment handling.
// A simpler variant pre-packs weights to aligned layout during upload.

// For now, the GPU path uses the F32 GEMV placeholder.
// The Q8_0 shader needs glslangValidator for SPIR-V compilation.
// The architecture is ready: upload Q8_0 weights as raw byte buffers,
// dispatch one workgroup per output row, 256 threads cooperatively reduce.

const glslDS4GemvQ8_0 = `
#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) readonly buffer Act { float activation[]; };
layout(set=0, binding=1) readonly buffer Wt  { float weight[]; };
layout(set=0, binding=2) writeonly buffer Out { float output[]; };
layout(push_constant) uniform P { uint inDim; uint outDim; };
shared float sdata[256];
void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    if (row >= outDim) return;
    float sum = 0.0;
    for (uint i = tid; i < inDim; i += 256)
        sum += weight[row * inDim + i] * activation[i];
    sdata[tid] = sum;
    barrier();
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0) output[row] = sdata[0];
}
`
