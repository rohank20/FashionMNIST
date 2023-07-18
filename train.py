import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from dataset_processing import load_data
from model import get_model

exp_name = 'initial_exp'


def train(train_images, train_labels, test_images, test_labels, epochs=15):
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    result = model.fit(
        x=train_images, y=train_labels,
        validation_data=(test_images, test_labels),
        epochs=epochs
    )

    return model, result


def evaluate_training(result):
    history_frame = pd.DataFrame(result.history)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot();
    plt.subplot()
    plt.show()
    val_accuracy = history_frame.val_accuracy.iloc[[-1]]
    print(f'Last validation accuracy: {val_accuracy}')


def save_model(model, exp_name):
    path = f'saved_models/{exp_name}'
    model.save(path)
    print(f'Model saved at path: {path}')


def main():
    train_images, train_labels, test_images, test_labels, class_list = load_data()
    model, result = train(train_images, train_labels, test_images, test_labels)
    evaluate_training(result)
    # save_model(model, exp_name)


if __name__ == '__main__':
    main()
