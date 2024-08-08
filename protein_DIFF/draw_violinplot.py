import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取CSV文件
df = pd.read_csv('assemble_at_end_results_test980.csv')

# 假设CSV文件包含如下列：Num, recovery, variance, perplexity, standard_error, recoveries
# 将'recoveries'列的字符串转换为列表
df['recoveries'] = df['recoveries'].apply(eval)

# 将数据转换为适合绘制箱型图的格式
data = []
for index, row in df.iterrows():
    num = row['Num']
    for recovery in row['recoveries']:
        data.append({'Num': num, 'Recovery': recovery})

# 转换为DataFrame
plot_df = pd.DataFrame(data)

# 计算 num=1 时的平均恢复率作为基准线
baseline = plot_df[plot_df['Num'] == 1]['Recovery'].mean()

# 绘制箱型图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Num', y='Recovery', data=plot_df)

# 添加基准线
plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline (Num=1): {baseline:.2f}')

# 设置图表标题和标签

plt.xlabel('Number of Self-Ensembles')
plt.ylabel('Recovery Rate')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.savefig('box_plot_with_baseline.pdf', format='pdf')
plt.show()