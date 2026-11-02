import matplotlib.pyplot as plt
import seaborn as sns


def create_histogram(df, column):

    fig, ax = plt.subplots()

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    return fig


def create_scatter(df, x_column, y_column):

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(f"{x_column} vs {y_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    return fig


def create_bar_chart(df, column):

    counts = df[column].value_counts().head(20)

    fig, ax = plt.subplots()

    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        ax=ax
    )

    ax.set_title(f"Count of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    plt.xticks(rotation=45)

    return fig


def create_heatmap(df):

    numerical_df = df.select_dtypes(include=["number"])

    correlation = numerical_df.corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")

    return fig