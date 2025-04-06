笔者加了点注释。

m、n 是数学上的逻辑矩阵在共享内存中的行列数，单位是元素个数。

stride_m、stride_n 是共享内存中行列的 stride，单位是元素个数。

## Usage

首先在本项目根目录运行 `cmake .`

然后在 `examples/cute_swizzle_preview` 目录下运行 `make`

然后，可以直接运行笔者准备的脚本 `run_cute_swizzle.sh`，可以在这个脚本中调整默认的参数。脚本会自动生成 latex 文件并编译为 pdf。

生成的 pdf 中，矩阵以逻辑 layout 呈现，每个格子代表一个元素，格子上的数字代表这个元素在共享内存中的 bank id。这里认为共享内存有 32 个 bank，bank id 从 0 到 31。
