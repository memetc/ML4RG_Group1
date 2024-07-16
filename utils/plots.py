import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score

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


def plot_predictions_vs_labels(predictions, labels, title="Predictions vs Labels", filename=None):
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
    if filename:
        plt.savefig(filename)
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


def plot_boxplot_predictions_vs_labels(predictions, labels, ids, by_label, filename=None):
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

    df_combined[by_label] = pd.Categorical(df_combined[by_label], categories=sorted(df_combined[by_label].unique()), ordered=True)


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
    plt.xticks(rotation=45)
    plt.grid(True)
    if filename:
        plt.savefig(filename)
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
    predictions, labels, title="Predictions vs Labels", gridsize=10, mincnt=5, filename=None
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
    if filename:
        plt.savefig(filename)
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


def plot_r2_scores_by_subgroup(X_test, y_test, xgb_model, subgroup_i, subgroup_mapping, subgroup_name):
    """
    Calculate and plot R-squared scores by subgroups.

    Parameters:
    - X_test (array-like or DataFrame): Test features.
    - y_test (array-like or Series): True test labels.
    - xgb_model (fitted model): Trained XGBoost model.
    - subgroup_mapping (dict): Dictionary mapping subgroup IDs to descriptive names.

    Returns:
    - None
    """

    # Convert to pandas DataFrame for easier manipulation if not already
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    # The subgroup identifiers
    subgroup_ids = X_test.iloc[:, subgroup_i]  # Assuming the first column contains subgroup IDs

    # Make predictions
    test_predictions = xgb_model.predict(X_test)

    # Create a DataFrame with predictions, true values, and subgroup IDs
    results_df = pd.DataFrame({
        'subgroup': subgroup_ids,
        'true': y_test,
        'predicted': test_predictions
    })

    # Calculate R-squared for each subgroup
    subgroup_r2_scores = results_df.groupby('subgroup').apply(lambda df: r2_score(df['true'], df['predicted']))

    # Convert the results to a DataFrame for easier plotting
    subgroup_r2_scores_df = pd.DataFrame(subgroup_r2_scores, columns=['R-squared']).reset_index()

    # Map the subgroup IDs to descriptive names using the mapping dictionary
    subgroup_r2_scores_df['subgroup'] = subgroup_r2_scores_df['subgroup'].map(subgroup_mapping)

    # Plot the R-squared scores for each subgroup
    plt.figure(figsize=(10, 6))
    sns.barplot(x='subgroup', y='R-squared', data=subgroup_r2_scores_df, palette='viridis')
    plt.xlabel(subgroup_name)
    plt.ylabel('R-squared')
    plt.title(f'R-squared Scores by {subgroup_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(xgb_model, feature_names=None, importance_type='weight', top_n=None):
    """
    Plot the feature importance of an XGBoost model.

    Parameters:
    - xgb_model (xgb.Booster or xgb.XGBModel): Trained XGBoost model.
    - importance_type (str): Importance type to plot. Options: 'weight', 'gain', 'cover'. Default is 'weight'.

    Returns:
    - None
    """

    # Get the feature importance score
    if isinstance(xgb_model, xgb.XGBModel):
        booster = xgb_model.get_booster()
    elif isinstance(xgb_model, xgb.Booster):
        booster = xgb_model
    else:
        raise ValueError("Unsupported model type")

    feature_importance = booster.get_score(importance_type=importance_type)

    # Convert to a DataFrame for easier plotting
    importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance'])
    # Replace feature indices with actual names

    if feature_names:
        importance_df['Feature'] = importance_df['Feature'].apply(lambda x: feature_names[int(x[1:])])

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance ({importance_type})')
    plt.yticks(rotation=45)  # Rotate the y-axis labels
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.show()


def calculate_r2_xg(xgb_model, X_tabular, y_test, species_col, condition_col):
    # Predict using the provided XGBoost model
    test_predictions = xgb_model.predict(X_tabular)
    r2_results = []

    # Create a dataframe with predictions and actual values
    results_df = X_tabular.copy()
    results_df['predicted'] = test_predictions
    results_df['actual'] = y_test
    results_df[species_col] = X_tabular[species_col].values
    results_df[condition_col] = X_tabular[condition_col].values

    species_conditions = results_df[[species_col, condition_col]].drop_duplicates()

    for index, row in species_conditions.iterrows():
        species = row[species_col]
        condition = row[condition_col]

        subset_df = results_df[(results_df[species_col] == species) & (results_df[condition_col] == condition)]

        if len(subset_df) > 1:  # Need at least two samples to calculate R2
            r2 = r2_score(subset_df['actual'], subset_df['predicted'])
            r2_results.append([species, condition, r2])
        else:
            r2_results.append([species, condition, None])

    r2_df = pd.DataFrame(r2_results, columns=[species_col, condition_col, 'R2'])
    return r2_df.pivot(index=species_col, columns=condition_col, values='R2')


def calculate_r2_cnn(predictions, labels, species_ids, stress_ids):
    r2_results = []

    # Create a dataframe with predictions and actual values
    results_df = pd.DataFrame({
        'predicted': predictions,
        'actual': labels,
        'species_id': species_ids,
        'condition': stress_ids
    })

    species_conditions = results_df[['species_id', 'condition']].drop_duplicates()

    for index, row in species_conditions.iterrows():
        species = row['species_id']
        condition = row['condition']

        subset_df = results_df[(results_df['species_id'] == species) & (results_df['condition'] == condition)]

        if len(subset_df) > 1:  # Need at least two samples to calculate R2
            r2 = r2_score(subset_df['actual'], subset_df['predicted'])
            r2_results.append([species, condition, r2])
        else:
            r2_results.append([species, condition, None])

    r2_df = pd.DataFrame(r2_results, columns=['species_id', 'condition', 'R2'])
    return r2_df.pivot(index='species_id', columns='condition', values='R2')

def plot_r2_values(r2_df, species_mapping, condition_mapping, filename=None):
    r2_df = r2_df.rename(index=species_mapping, columns=condition_mapping)

    plt.figure(figsize=(10, 15))
    sns.heatmap(r2_df, annot=True, cmap='Greens', linewidths=.5, linecolor='black', cbar_kws={'label': 'R2 Value'})
    plt.title('R2 Values for Different Species and Conditions')
    plt.xlabel('Condition')
    plt.ylabel('Species')
    if filename:
        plt.savefig(filename)
    plt.show()





