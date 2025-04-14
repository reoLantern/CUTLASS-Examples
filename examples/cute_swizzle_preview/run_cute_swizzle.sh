#!/bin/bash

# 默认参数
M=32
N=8
STRIDE_M=8
STRIDE_N=1
ELEMENT_SIZE=2            # 元素大小（字节）
SWIZZLE_NUM_MASK_BITS=0   # B
SWIZZLE_NUM_BASE=4        # M
SWIZZLE_NUM_SHIFT=3       # S
LATEX_FILE_PATH="shared_memory_bank_ids.tex"
MODE="bank+address"

# 如果传递了参数，覆盖默认值
while [[ $# -gt 0 ]]; do
    case "$1" in
        --m)
            M="$2"
            shift 2
            ;;
        --n)
            N="$2"
            shift 2
            ;;
        --stride_m)
            STRIDE_M="$2"
            shift 2
            ;;
        --stride_n)
            STRIDE_N="$2"
            shift 2
            ;;
        --element_size)
            ELEMENT_SIZE="$2"
            shift 2
            ;;
        --swizzle_num_mask_bits)
            SWIZZLE_NUM_MASK_BITS="$2"
            shift 2
            ;;
        --swizzle_num_base)
            SWIZZLE_NUM_BASE="$2"
            shift 2
            ;;
        --swizzle_num_shift)
            SWIZZLE_NUM_SHIFT="$2"
            shift 2
            ;;
        --latex_file_path)
            LATEX_FILE_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --m=<int>                         Matrix on shared memory M dimension"
            echo "  --n=<int>                         Matrix on shared memory N dimension"
            echo "  --stride_m=<int>                  Matrix on shared memory M stride"
            echo "  --stride_n=<int>                  Matrix on shared memory N stride"
            echo "  --element_size=<int>              Element size in bytes"
            echo "  --swizzle_num_mask_bits=<int>     Number of swizzle mask bits"
            echo "  --swizzle_num_base=<int>          Number of swizzle base bits"
            echo "  --swizzle_num_shift=<int>         Number of swizzle shift bits"
            echo "  --latex_file_path=<string>        LaTeX file path"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 打印传入的配置
echo "Running with the following configuration:"
echo "m: $M"
echo "n: $N"
echo "stride_m: $STRIDE_M"
echo "stride_n: $STRIDE_N"
echo "element_size: $ELEMENT_SIZE"
echo "swizzle_num_mask_bits: $SWIZZLE_NUM_MASK_BITS"
echo "swizzle_num_base: $SWIZZLE_NUM_BASE"
echo "swizzle_num_shift: $SWIZZLE_NUM_SHIFT"
echo "latex_file_path: $LATEX_FILE_PATH"
echo "mode: $MODE"

# 运行命令
./cute_swizzle_preview \
    --m=$M --n=$N --stride_m=$STRIDE_M --stride_n=$STRIDE_N \
    --element_size=$ELEMENT_SIZE --swizzle_num_mask_bits=$SWIZZLE_NUM_MASK_BITS \
    --swizzle_num_base=$SWIZZLE_NUM_BASE --swizzle_num_shift=$SWIZZLE_NUM_SHIFT \
    --latex_file_path=$LATEX_FILE_PATH \
    --mode=$MODE \
    && latexmk -pdf -shell-escape ./$LATEX_FILE_PATH  # you can replace -pdf with -lualatex to use lualatex
