# 使用强化学习完成井字棋
## 总览
1. 首先我们要编写一个简单的 井字棋 的游戏，准备使用 gym 进行编写，实现 人人对战的功能 并进行 图像的渲染，能够在一个窗口中可视化出来。



## 制作井字棋环境
参考链接： 
https://www.cnblogs.com/wsy950409/p/15647914.html
提供了一个基于gym可视化的 $3 \times 3$ 的井字棋环境
https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/tree/master/chapter01
提供了一个基于命令行的 $N \times N$ 的井字棋环境

我们首先要做的是将他们给结合起来，制作一个 一个基于gym可视化的 $N \times N$ 的井字棋环境

