import pandas as pd 
from sklearn.model_selection import train_test_split

inp_file = "metadata_94_labels.csv"

df = pd.read_csv(inp_file)

#70, 30
train_df, temp_df = train_test_split(
    df, test_size=0.3, random_state=42, shuffle=True,)


#30 = 15+15
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, shuffle=True)


train_df.to_csv("train_split.csv", index=False)
val_df.to_csv("val_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)

print("split completed!")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))