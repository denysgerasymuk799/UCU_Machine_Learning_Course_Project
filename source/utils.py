import matplotlib.pyplot as plt


def run_sequence_plot(df, y_column, title, from_datetime=None, to_datetime=None):
    if not from_datetime:
        from_datetime = df.index.min()
    if not to_datetime:
        to_datetime = df.index.max()

    plt.plot(df[from_datetime: to_datetime].index, df[y_column][from_datetime: to_datetime].values,
             color='blue', label="Radiation")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
