/* encoder/dct8x8.c - 修复版 */
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 
 * 核心 DCT 函数 
 * 修改为支持输入(input)和输出(output)分离
 */
void perform_dct(const double *input, double *output) {
    double temp[64];
    double cu, cv, sum;
    int u, v, x, y;

    for (u = 0; u < 8; ++u) {
        for (v = 0; v < 8; ++v) {
            sum = 0.0;
            if (u == 0) cu = 1.0 / sqrt(2.0); else cu = 1.0;
            if (v == 0) cv = 1.0 / sqrt(2.0); else cv = 1.0;

            for (x = 0; x < 8; ++x) {
                for (y = 0; y < 8; ++y) {
                    // 从 input 读取
                    sum += input[x * 8 + y] * 
                           cos((2 * x + 1) * u * M_PI / 16.0) * 
                           cos((2 * y + 1) * v * M_PI / 16.0);
                }
            }
            temp[u * 8 + v] = 0.25 * cu * cv * sum;
        }
    }

    // 写入 output
    for (int i = 0; i < 64; ++i) {
        output[i] = temp[i];
    }
}

/* 
 * 批量处理函数
 * 必须匹配 Python: (double* input, double* output, int n)
 */
void dct8x8_batch(double *input_blocks, double *output_blocks, int n) {
    for (int i = 0; i < n; ++i) {
        // 输入和输出指针同时偏移
        perform_dct(input_blocks + i * 64, output_blocks + i * 64);
    }
}

/* 兼容性保留: 原地处理 (In-place) */
void dct8x8(double *block) {
    perform_dct(block, block);
}