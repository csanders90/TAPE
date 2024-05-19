import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
data = np.load('/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns/data_seal.npz')
data1 = data['pos_pred']
data2 = data['neg_pred']


# 创建图形和主轴
fig, ax = plt.subplots(figsize=(10, 6))

# 创建左侧直方图
ax_left = ax.twiny()  # 创建一个共享y轴的次坐标轴
ax_left.hist(data1, bins=30, orientation='horizontal', color='blue', edgecolor='black', alpha=0.7)
ax_left.invert_xaxis()  # 反转x轴，使柱状图朝向中心
ax_left.set_xlabel('Count (Left Histogram)')
ax_left.set_ylabel('Frequency')
ax_left.set_title('Histograms Facing Each Other')

# 创建右侧直方图
ax.hist(data2, bins=30, orientation='horizontal', color='green', edgecolor='black', alpha=0.7)
ax.set_xlabel('Count (Right Histogram)')
ax.yaxis.tick_right()  # y轴刻度标签显示在右侧
ax.yaxis.set_label_position("right")

# 调整布局
plt.tight_layout()


# 显示图形
plt.savefig('/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/core/gcns/hah.png')
