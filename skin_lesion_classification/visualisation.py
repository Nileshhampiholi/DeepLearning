import matplotlib.pyplot as plt
import seaborn as sns


class Visualisation:
    @staticmethod
    def plot_loses(training, validation, label="", num=0):
        plt.figure(num=num, figsize=[8, 8], dpi=300)
        plt.plot(range(1, len(training) + 1), training, 'y', label='Training ' + label)
        plt.plot(range(1, len(training) + 1), validation, 'r', label='Validation ' + label)
        plt.title('Training and validation ' + label)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_heat_map(confusion_matrix):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.set(font_scale=1.6)
        sns.heatmap(confusion_matrix, annot=True, linewidths=.5, ax=ax)
