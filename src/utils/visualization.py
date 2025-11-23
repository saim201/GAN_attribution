import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_cm(cm, title, save_path, class_names):
    plt.figure(figsize=(4.2, 3.8))
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, square=True)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()
