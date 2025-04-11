#pragma once

#include "cutlass/device_kernel.h"
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;


// 可变参数的打印函数模板
template <typename PrintFunc, typename... Args>
CUTE_HOST_DEVICE void custom_print_func(const char *prefix, PrintFunc printFunc, Args &&...args)
{
    printf("%s", prefix);
    printFunc(std::forward<Args>(args)...);
}

template <typename LayoutT> CUTE_HOST_DEVICE void custom_print_layout(const char *prefix, const LayoutT &lay_)
{
    printf("%s", prefix);
    print(lay_);
    print("\n");
}

// -------------------- print_data 函数，模仿 cute::print_layout() -------------------
template <class Layout, class DT> CUTE_HOST_DEVICE void print_data(Layout const &layout, DT const *data)
{
    CUTE_STATIC_ASSERT_V(rank(layout) == cute::Int<2>{});

    int idx_width = num_digits(cosize(layout)) + 2;
    const char *delim = "+-----------------------";

    print(layout);
    print("\n");

    printf("Data:\n");

    // Column indices
    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        printf("  %*d ", idx_width - 2, n);
    }
    printf("\n");

    // Print out A m-by-n
    for (int m = 0; m < cute::size<0>(layout); ++m)
    {
        printf("    ");
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            printf("%.*s", idx_width + 1, delim);
        }
        printf("+\n");

        printf("%2d  ", m);
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            int offset = layout(m, n);
            printf("| %*g ", idx_width - 2, float(data[offset]));
        }
        printf("|\n");
    }
    // Footer
    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        printf("%.*s", idx_width + 1, delim);
    }
    printf("+\n");
}

// -------------------- 打印 [地址偏移, 实际数据] 格式  -------------------
template <class Layout, class DT> CUTE_HOST_DEVICE void print_data_with_layout(Layout const &layout, DT const *data)
{
    CUTE_STATIC_ASSERT_V(rank(layout) == cute::Int<2>{});

    // 原来每个单元格预留宽度
    int idx_width = num_digits(cosize(layout)) + 2;
    // 现在每个单元格需打印 "[<offset>, <data>]"，其中两个数字各占 idx_width 个字符，加上
    // 字符 '['、", "、']' 总共需要 1 + 2 + 1 = 4 个额外字符，因此新的单元格宽度：
    int cell_width = 2 * idx_width + 4;
    // 为了在左右两边额外保留空格，再加 1 个空格，最终设为：
    cell_width += 1;

    // 用于打印水平分隔线，注意我们将其长度调整为 cell_width + 1
    const char *delim = "+-----------------------";

    // 打印 layout 信息
    print(layout);
    print("\n");

    // 增加表头说明
    printf("Data (each cell is printed as [offset, data]):\n");

    // 打印列索引，按照新的 cell_width 格式打印
    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        // 这里我们以 cell_width-2 的宽度打印列号（使其在单元格内相对居中）
        printf("  %*d ", cell_width - 2, n);
    }
    printf("\n");

    // 对每一行进行打印
    for (int m = 0; m < cute::size<0>(layout); ++m)
    {
        // 打印行内的水平分割线
        printf("    ");
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            printf("%.*s", cell_width + 1, delim);
        }
        printf("+\n");

        // 打印行索引
        printf("%2d  ", m);
        // 每个单元格打印 [地址偏移, 实际数据]
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            int offset = layout(m, n);
            // 格式说明：
            //   "["
            //   然后用 idx_width 宽度打印地址偏移
            //   然后打印 ", "
            //   然后用 idx_width 宽度打印数据（这里将数据转换为 float 显示）
            //   再打印 "]"
            printf("| [%*d, %*g] ", idx_width, offset, idx_width, float(data[offset]));
        }
        printf("|\n");
    }
    // 打印表格最后一行的分割线
    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        printf("%.*s", cell_width + 1, delim);
    }
    printf("+\n");
}
