import numpy as np

from neural_network import NeuralNetwork
from data_loader import MnistDataloader

# Hyperparameters
HIDDEN_LAYER_SIZES = [256, 128, 64, 32]
EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.01
RANDOM_SEED = 0
NUM_CLASSES = 10


def one_hot_encode(labels, num_classes):
    labels_array = np.asarray(labels, dtype=np.int64)
    encoded = np.zeros((labels_array.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels_array.shape[0]), labels_array] = 1.0
    return encoded


def preprocess_images(images):
    images_array = np.asarray(images, dtype=np.float32)
    if images_array.ndim == 3:
        images_array = images_array.reshape(images_array.shape[0], -1)
    return images_array / 255.0


def train(network, images, labels_int, targets_one_hot, epochs, batch_size, learning_rate):
    num_samples = images.shape[0]
    for epoch in range(epochs):
        progress_percent = int(((epoch) / epochs) * 100)
        print(f"Progress: {progress_percent}% (before epoch {epoch + 1})")
        permutation = np.random.permutation(num_samples)
        total_loss = 0.0
        correct_predictions = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = permutation[start:end]
            effective_batch = end - start

            grad_w = [np.zeros_like(weight) for weight in network.all_weights]
            grad_b = [np.zeros_like(bias) for bias in network.all_biases]

            for idx in batch_indices:
                x_column = images[idx].reshape(-1, 1)
                target_column = targets_one_hot[idx].reshape(-1, 1)

                probabilities, all_f, all_h = network.forwardPassForOneInput(x_column)
                total_loss += network.cross_entropy_loss(probabilities, target_column)
                dl_dw, dl_db = network.backward_pass(all_f, all_h, target_column)

                for layer in range(len(grad_w)):
                    grad_w[layer] += dl_dw[layer]
                    grad_b[layer] += dl_db[layer]

                prediction = int(np.argmax(probabilities))
                correct_predictions += int(prediction == labels_int[idx])

            for layer in range(len(network.all_weights)):
                grad_w[layer] /= effective_batch
                grad_b[layer] /= effective_batch
                network.all_weights[layer] -= learning_rate * grad_w[layer]
                network.all_biases[layer] -= learning_rate * grad_b[layer]

        average_loss = total_loss / num_samples
        accuracy = correct_predictions / num_samples
        accuracy_pct = accuracy * 100.0
        epoch_progress = int(((epoch + 1) / epochs) * 100)
        print(
            f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}, "
            f"accuracy: {accuracy_pct:.2f}% ({epoch_progress}%)"
        )

    print("Progress: 100% (training complete)")

def evaluate(network, images, labels_int):
    correct = 0
    for idx in range(images.shape[0]):
        x_column = images[idx].reshape(-1, 1)
        probabilities, _, _ = network.forwardPassForOneInput(x_column)
        prediction = int(np.argmax(probabilities))
        correct += int(prediction == labels_int[idx])
    accuracy = correct / images.shape[0]
    return accuracy


def main():
    np.random.seed(RANDOM_SEED)

    input_path = './mnistData/'
    training_images_filepath = input_path + 'train-images-idx3-ubyte'
    training_labels_filepath = input_path + 'train-labels-idx1-ubyte'
    test_images_filepath = input_path + 't10k-images-idx3-ubyte'
    test_labels_filepath = input_path + 't10k-labels-idx1-ubyte'

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    x_train_processed = preprocess_images(x_train)
    x_test_processed = preprocess_images(x_test)

    y_train_int = np.asarray(y_train, dtype=np.int64)
    y_test_int = np.asarray(y_test, dtype=np.int64)

    y_train_one_hot = one_hot_encode(y_train_int, NUM_CLASSES)

    input_size = x_train_processed.shape[1]
    network = NeuralNetwork(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        inputLayerSize=input_size,
        OutputLayerSize=NUM_CLASSES,
        seed=RANDOM_SEED,
    )

    train(
        network,
        x_train_processed,
        y_train_int,
        y_train_one_hot,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    test_accuracy = evaluate(network, x_test_processed, y_test_int)
    print(f"Test accuracy: {test_accuracy * 100.0:.2f}%")


if __name__ == '__main__':
    main()
