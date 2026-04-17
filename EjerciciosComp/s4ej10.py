import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.DataFrame(
    {
        "color": ["rojo", "azul", "verde", "rojo", "verde"],
        "talla": ["S", "M", "L", "S", "M"],
    }
)

le_color = LabelEncoder()
le_talla = LabelEncoder()
df_label = df.copy()
df_label["color"] = le_color.fit_transform(df_label["color"])
df_label["talla"] = le_talla.fit_transform(df_label["talla"])

df_dummies = pd.get_dummies(df, columns=["color", "talla"], dtype=int)

one_hot = OneHotEncoder(sparse_output=False)
one_hot_data = one_hot.fit_transform(df)
one_hot_cols = one_hot.get_feature_names_out(df.columns)
df_onehot_sklearn = pd.DataFrame(one_hot_data, columns=one_hot_cols)

print("DataFrame original:\n", df)
print("\nLabel Encoding:\n", df_label)
print("\nOne-Hot con get_dummies:\n", df_dummies)
print("\nOne-Hot con sklearn:\n", df_onehot_sklearn)
