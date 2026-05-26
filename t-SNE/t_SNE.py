import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
# Import class của bạn để lấy data
from src.preprocessing.preprocessing import Preprocessing

# 1. Khởi tạo và lấy dữ liệu X, y gốc
preprocessor = Preprocessing()
data = preprocessor.run()
X = data.drop('target', axis=1)
y = data['target']

# 2. Sao chép data để xử lý riêng cho t-SNE
cat_cols = ['tourney_name', 'surface', 'tourney_level', 'round']
X_tsne = X.copy()

# 3. Mã hóa Ordinal để t-SNE tính được khoảng cách hình học
encoder = OrdinalEncoder()
X_tsne[cat_cols] = encoder.fit_transform(X_tsne[cat_cols])

# 4. Chuẩn hóa thang đo (Scale)
scaler = StandardScaler()
X_tsne_scaled = scaler.fit_transform(X_tsne)

# 5. Chạy t-SNE giảm về 2 chiều (n_jobs=-1 để tận dụng CPU Mac)
tsne = TSNE(n_components=2, perplexity=50, random_state=42, n_jobs=-1, max_iter=1000)
X_embedded = tsne.fit_transform(X_tsne_scaled)

# 6. Tạo DataFrame kết quả phục vụ vẽ biểu đồ
df_tsne = pd.DataFrame(X_embedded, columns=['t-SNE Component 1', 't-SNE Component 2'])
df_tsne['target'] = y.values

# 7. Vẽ biểu đồ bằng Seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='t-SNE Component 1',
    y='t-SNE Component 2',
    hue='target',
    palette='coolwarm',
    data=df_tsne,
    alpha=0.5,
    s=10
)

plt.title('t-SNE Visualization of Tennis Match Raw Data', fontsize=14)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Target (0: Lose / 1: Win)')
plt.grid(True, alpha=0.3)
plt.show()