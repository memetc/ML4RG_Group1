import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# plot the losses
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.show()

    log_train_losses = np.log(train_losses)
    log_val_losses = np.log(val_losses)
    plt.plot(log_train_losses, label='Log Train Loss')
    plt.plot(log_val_losses, label='Log Validation Loss')
    plt.legend()
    plt.show()

def plot_predictions_vs_labels(predictions, labels, title="Predictions vs Labels"):
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, predictions, alpha=0.5)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
    plt.xlabel('True Labels')
    plt.ylabel('Predictions')
    plt.title(title)
    plt.grid(True)
    plt.show()



def plot_predictions_vs_labels_by_species(predictions, labels, species_ids, title="Predictions vs Labels by Species"):
    df = pd.DataFrame({'SpeciesID': species_ids, 'TrueLabel': labels, 'Prediction': predictions})
    species_groups = df.groupby('SpeciesID')
    
    for species_id, group in species_groups:
        plt.figure(figsize=(10, 6))
        plt.scatter(group['TrueLabel'], group['Prediction'], alpha=0.5)
        plt.plot([group['TrueLabel'].min(), group['TrueLabel'].max()], 
                 [group['TrueLabel'].min(), group['TrueLabel'].max()], 'r--', lw=2)
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.title(f'{title} - Species ID: {species_id}')
        plt.grid(True)
        plt.show()


def plot_boxplot_predictions_vs_labels(predictions, labels, ids, by_label):
    df_predictions = pd.DataFrame({
        by_label: ids,
        'Value': predictions,
        'Type': 'Prediction'
    })
    df_labels = pd.DataFrame({
        by_label: ids,
        'Value': labels,
        'Type': 'Label'
    })

    df_combined = pd.concat([df_predictions, df_labels])

    plt.figure(figsize=(12, 8))
    sns.boxplot(x=by_label, y='Value', hue='Type', data=df_combined, palette=['#1f77b4', '#ff7f0e'])

    plt.title(f"Predictions and Labels by {by_label}")
    plt.xlabel(by_label)
    plt.ylabel('Value')
    plt.legend(title='Type')
    plt.grid(True)
    plt.show()