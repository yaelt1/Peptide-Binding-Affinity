import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'df' is your DataFrame containing the Pearson correlation coefficients
# df = pd.read_csv("/Users/yaeltzur/Desktop/uni/third_yaer/סדנה/results_cnn.csv", index_col=0)


def heat_map(df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Pearson Correlation Coefficients CNN')
    plt.show()
    
    
if __name__=="__main__":
    df = pd.read_csv("/Users/yaeltzur/Desktop/uni/third_yaer/סדנה/results_cnn_new.csv", index_col=0)
    df_sorted_rows = df.sort_index()

    # Sort by columns index names
    df_sorted_both = df_sorted_rows[sorted(df_sorted_rows.columns)]

    heat_map(df_sorted_both)

    
