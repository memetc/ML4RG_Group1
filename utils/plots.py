import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# plot the losses
def plot_losses(train_losses, val_losses):
    """
    Plot training and validation losses over epochs.

    This function plots the training and validation losses over the epochs.
    It also plots the logarithm of these losses for better visualization of trends.

    Parameters:
    - train_losses (list): List of training losses.
    - val_losses (list): List of validation losses.

    Returns:
    - None
    """
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.show()

    log_train_losses = np.log(train_losses)
    log_val_losses = np.log(val_losses)
    plt.plot(log_train_losses, label="Log Train Loss")
    plt.plot(log_val_losses, label="Log Validation Loss")
    plt.legend()
    plt.show()


def plot_predictions_vs_labels(predictions, labels, title="Predictions vs Labels"):
    """
    Plot a scatter plot of predictions vs. true labels.

    This function creates a scatter plot to visualize the correlation between the predicted values
    and the true labels. It also plots the identity line representing perfect predictions.

    Parameters:
    - predictions (array-like): The predicted values.
    - labels (array-like): The true labels.
    - title (str, optional): The title of the plot. Default is "Predictions vs Labels".

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, labels, alpha=0.5)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], "r--", lw=2)
    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_predictions_vs_labels_by_species(
    predictions, labels, species_ids, title="Predictions vs Labels by Species"
):
    """
    Plot scatter plots of predictions vs. true labels, grouped by species.

    This function creates scatter plots to visualize the correlation between the predicted values
    and the true labels, grouped by species ID. Each species gets its own plot.

    Parameters:
    - predictions (array-like): The predicted values.
    - labels (array-like): The true labels.
    - species_ids (array-like): The species IDs corresponding to the predictions and labels.
    - title (str, optional): The title of the plot. Default is "Predictions vs Labels by Species".

    Returns:
    - None
    """
    df = pd.DataFrame(
        {"SpeciesID": species_ids, "TrueLabel": labels, "Prediction": predictions}
    )
    species_groups = df.groupby("SpeciesID")

    for species_id, group in species_groups:
        plt.figure(figsize=(10, 6))
        plt.scatter(group["Prediction"], group["TrueLabel"], alpha=0.5)
        plt.plot(
            [group["TrueLabel"].min(), group["TrueLabel"].max()],
            [group["TrueLabel"].min(), group["TrueLabel"].max()],
            "r--",
            lw=2,
        )
        plt.ylabel("True Labels")
        plt.xlabel("Predictions")
        plt.title(f"{title} - Species ID: {species_id}")
        plt.grid(True)
        plt.show()


def filter_outliers(df, by_label):
    """
    Remove outliers from the dataset using the IQR method.

    Parameters:
    - df (DataFrame): The combined DataFrame containing predictions and labels.
    - by_label (str): The label by which to group the data.

    Returns:
    - DataFrame: The DataFrame with outliers removed.
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df.groupby(by_label)['Value'].quantile(0.25)
    Q3 = df.groupby(by_label)['Value'].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    filtered_df = df[
        ~((df['Value'] < lower_bound[df[by_label]].values) | (df['Value'] > upper_bound[df[by_label]].values))]

    return filtered_df


def plot_boxplot_predictions_vs_labels(predictions, labels, ids, by_label):
    """
    Plot box plots of predictions and true labels, grouped by a specified label.

    This function creates box plots to visualize the distribution of predicted values and true labels,
    grouped by a specified label (e.g., species ID).

    Parameters:
    - predictions (array-like): The predicted values.
    - labels (array-like): The true labels.
    - ids (array-like): The IDs corresponding to the predictions and labels (e.g., species IDs).
    - by_label (str): The label by which to group the box plots (e.g., 'SpeciesID').

    Returns:
    - None
    """
    df_predictions = pd.DataFrame(
        {by_label: ids, "Value": predictions, "Type": "Prediction"}
    )
    df_labels = pd.DataFrame({by_label: ids, "Value": labels, "Type": "Label"})

    df_combined = pd.concat([df_predictions, df_labels])
    df_combined = filter_outliers(df_combined, by_label)


    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x=by_label,
        y="Value",
        hue="Type",
        data=df_combined,
        palette=["#1f77b4", "#ff7f0e"],
    )

    plt.title(f"Predictions and Labels by {by_label}")
    plt.xlabel(by_label)
    plt.ylabel("Value")
    plt.legend(title="Type")
    plt.grid(True)
    plt.show()


def plot_density_predictions_vs_labels(
    predictions, labels, title="Predictions vs Labels"
):
    """
    Plot a density plot of predictions vs. true labels with inverse normalization.

    This function creates a density plot to visualize the correlation between the predicted values
    and the true labels. It also plots the identity line representing perfect predictions. The values
    are inverse normalized using the exponential function to revert the logarithmic transformation.

    Parameters:
    - predictions (array-like): The predicted values.
    - labels (array-like): The true labels.
    - title (str, optional): The title of the plot. Default is "Predictions vs Labels".

    Returns:
    - None
    """
    # Inverse normalize the values
    predictions = np.exp(predictions) - 1
    labels = np.exp(labels) - 1

    plt.figure(figsize=(10, 6))

    # Use seaborn to create a density plot
    sns.kdeplot(x=predictions, y=labels, cmap="Blues", shade=True, bw_adjust=0.5)

    # Plot the identity line (ideal predictions)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], "r--", lw=2)

    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_hexbin_predictions_vs_labels(
    predictions, labels, title="Predictions vs Labels", gridsize=10, mincnt=5
):
    """
    Plot a hexbin plot of predictions vs. true labels with inverse normalization.

    This function creates a hexbin plot to visualize the correlation between the predicted values
    and the true labels. It also plots the identity line representing perfect predictions. The values
    are inverse normalized using the exponential function to revert the logarithmic transformation.

    Parameters:
    - predictions (array-like): The predicted values.
    - labels (array-like): The true labels.
    - title (str, optional): The title of the plot. Default is "Predictions vs Labels".

    Returns:
    - None
    """

    plt.figure(figsize=(10, 6))

    # Create a hexbin plot
    hb = plt.hexbin(
        predictions,
        labels,
        gridsize=gridsize,
        cmap="Blues",
        mincnt=mincnt,
        extent=[labels.min(), labels.max(), labels.min(), labels.max()],
    )

    # Add a color bar
    cb = plt.colorbar(hb)
    cb.set_label("Counts")

    # Plot the identity line (ideal predictions)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], "r--", lw=2)

    plt.xlabel("Predictions")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_histogram(
    data_series,
    title="Histogram",
    xlabel="Value",
    ylabel="Frequency",
    bins=100,
    save=True,
):
    """
    Plots a histogram for a given Pandas Series.

    Parameters:
    - data_series: Pandas Series to plot.
    - title: Title of the histogram.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - bins: Number of bins in the histogram.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data_series.dropna(), bins=bins, edgecolor="k", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save:
        plt.savefig(f"{title}.png")

    plt.show()
