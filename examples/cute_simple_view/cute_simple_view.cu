#include <iostream>
#include <vector>

// 不需要 #include <cute/print.hpp>
#include <cute/tensor.hpp>
#include <cute/layout.hpp>

using namespace cute;

int main() {
    // ============================================================
    // 1. 定义全局内存（Global Memory）的布局
    // 假设全局矩阵尺寸为 16×16，采用行主序。
    // 行主序：逻辑坐标 (row, col)，其 stride 为 (16, 1)：
    //   - 对 row 增加 1，对应物理地址增加 16；
    //   - 对 col 增加 1，对应物理地址增加 1。
    constexpr int GlobalRows = 16, GlobalCols = 16;
    auto global_shape   = make_shape(Int<GlobalRows>{}, Int<GlobalCols>{});
    auto global_stride  = make_stride(Int<GlobalCols>{}, Int<1>{});
    auto global_layout  = make_layout(global_shape, global_stride);

    // 分配全局内存缓冲区，并用不同的值填充（例如用下标值方便观察）
    std::vector<float> global_buffer(GlobalRows * GlobalCols);
    for (int i = 0; i < GlobalRows * GlobalCols; i++) {
        global_buffer[i] = static_cast<float>(i);
    }
    // 用 tensor 封装（方便后续通过 layout() 计算偏移）
    auto global_tensor = make_tensor(global_buffer.data(), global_layout);

    std::cout << "----- Global Memory Layout -----" << std::endl;
    print_layout(global_layout);  // 打印出形状和 stride 信息
    std::cout << std::endl;

    // ============================================================
    // 2. 定义共享内存（Shared Memory）的布局
    // 这里我们选择拷贝一个 8×8 的 tile 到共享内存，
    // 并故意采用列主序存储，即逻辑尺寸 8×8，stride 为 (1, 8):
    //   - 对第 0 维增加 1，对应地址增加 1；
    //   - 对第 1 维增加 1，对应地址增加 8.
    constexpr int TileRows = 8, TileCols = 8;
    auto shared_shape   = make_shape(Int<TileRows>{}, Int<TileCols>{});
    auto shared_stride  = make_stride(Int<1>{}, Int<TileRows>{});
    auto shared_layout  = make_layout(shared_shape, shared_stride);

    // 分配共享内存缓冲区（在真实 GPU 程序中，这部分位于 __shared__ 内存中）
    std::vector<float> shared_buffer(TileRows * TileCols, 0);
    auto shared_tensor = make_tensor(shared_buffer.data(), shared_layout);

    std::cout << "----- Shared Memory Layout -----" << std::endl;
    print_layout(shared_layout);
    std::cout << std::endl;

    // ============================================================
    // 3. 模拟 Global -> Shared 内存拷贝过程
    // 假设我们拷贝的 tile 在全局矩阵中的起点坐标为 (4, 4)
    int tile_offset_row = 4;
    int tile_offset_col = 4;

    // 遍历 tile 内每个元素
    for (int r = 0; r < TileRows; ++r) {
        for (int c = 0; c < TileCols; ++c) {
            // 全局矩阵中的逻辑坐标
            auto global_coord = make_coord(tile_offset_row + r, tile_offset_col + c);
            // 计算全局内存中的线性索引，使用 operator()
            auto global_index = global_layout(global_coord);

            // 共享内存中的逻辑坐标
            auto shared_coord = make_coord(r, c);
            auto shared_index = shared_layout(shared_coord);

            // 拷贝数据
            shared_buffer[shared_index] = global_buffer[global_index];
        }
    }

    // ============================================================
    // 4. 打印拷贝后的共享内存内容（数据值）
    std::cout << "----- Shared Memory Contents after Global->Shared Copy -----" << std::endl;
    for (int r = 0; r < TileRows; ++r) {
        for (int c = 0; c < TileCols; ++c) {
            auto idx = shared_layout(make_coord(r, c));
            std::cout << shared_buffer[idx] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ============================================================
    // 5. 打印共享内存中每个数据的相对地址（物理 offset）
    std::cout << "----- Shared Memory Relative Offsets -----" << std::endl;
    for (int r = 0; r < TileRows; ++r) {
        for (int c = 0; c < TileCols; ++c) {
            auto offset = shared_layout(make_coord(r, c));
            std::cout << offset << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}
