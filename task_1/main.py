import pandas as pd
import matplotlib.pyplot as plt


def build_histogram_of_survival_by_age(train_dataset):
    plt.hist(
        x=[train_dataset[train_dataset['Survived'] == 1]['Age'], train_dataset[train_dataset['Survived'] == 0]['Age']],
        stacked=True, color=['b', 'r'], label=['Survived', 'Dead'])
    plt.title('Histogram')
    plt.xlabel('Age')
    plt.ylabel('Number of Passengers')
    plt.show()


def build_histogram_of_survival_by_class(train_dataset):
    plt.hist(x=[train_dataset[train_dataset['Survived'] == 1]['Pclass'], train_dataset[train_dataset['Survived'] == 0]['Pclass']],
             stacked=True, color=['b', 'r'], label=['Survived', 'Dead'])
    plt.title('Histogram')
    plt.xlabel('Class')
    plt.ylabel('Number of Passengers')
    plt.show()


if __name__ == '__main__':
    train_dataset = pd.read_csv("../dataset/train.csv")
    build_histogram_of_survival_by_age(train_dataset)
    build_histogram_of_survival_by_class(train_dataset)

