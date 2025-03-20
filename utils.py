import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy(history, title="Federated Learning Accuracy"):
    """Plots accuracy over federated training rounds."""
    rounds = range(1, len(history) + 1)
    sns.lineplot(x=rounds, y=history)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.show()
