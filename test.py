import pandas as pd
import numpy as np

# Tạo DataFrame theo yêu cầu
data = {
    'Attributes': ['high', 'low', 'open', 'close', 'avg', 'volume'],
    'Symbols': ['VCB'] * 6,
    '2020-01-02': [91.4, 89.7, 90.2, 90.8, 90.68, 386290.0],
    '2020-01-03': [91.8, 89.9, 91.5, 89.9, 90.81, 536130.0],
    '2020-01-04': [np.nan] * 6,
    '2020-01-05': [np.nan] * 6,
    '2020-01-06': [89.5, 87.5, 89.2, 87.5, 88.54, 880110.0],
    '2020-01-07': [87.9, 85.4, 87.0, 87.8, 86.54, 1013270.0],
    '2020-01-08': [87.9, 86.2, 86.9, 87.0, 86.97, 722280.0],
    '2020-01-09': [88.9, 87.7, 87.7, 88.7, 88.24, 722670.0],
    '2020-01-10': [90.4, 87.9, 88.7, 89.5, 88.66, 1526920.0],
    '2020-01-11': [np.nan] * 6,
}

df = pd.DataFrame(data)

# Chuyển vị DataFrame
df_transposed = df.transpose()

# Reset index để loại bỏ MultiIndex
df_transposed = df.reset_index(drop=True)

# Hiển thị DataFrame chuyển vị sau khi xóa MultiIndex
print(df_transposed)
