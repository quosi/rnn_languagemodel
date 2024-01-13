import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.utils import plot_model, get_file
import matplotlib.pyplot as plt


def get_text():
    path = 'data/Drei-Meister-Balzac-Dickens-Dostojewski_Stefan-Zweig_Projekt-Gutenberg_36389-8.txt'
    with open(path) as whole_text:
        text = whole_text.read().lower()
    return text


def preprocess_split(text, max_len, step):
    sentences, next_char = [], []
    for i in range(0, len(text) - max_len, step):
        sentences.append(text[i: i + max_len])
        next_char.append(text[i + max_len])
    char_lst = sorted(list(set(text)))
    char_dict = {char: char_lst.index(char) for char in char_lst}
    X = np.zeros((len(sentences), max_len, len(char_lst)), dtype=bool)
    y = np.zeros((len(next_char), len(char_lst)), dtype=bool)
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char_dict[char]] = 1
        y[i, char_dict[next_char[i]]] = 1
    return X, y, char_dict


def build_model(max_len, vocab_size):
    # build LSTM RNN model
    inputs = layers.Input(shape=(max_len, vocab_size))
    x = layers.LSTM(128)(inputs)
    output = layers.Dense(vocab_size, activation=tf.nn.softmax)(x)
    model1 = Model(inputs, output)
    model1.compile(optimizer='adam', loss='categorical_crossentropy')
    return model1


def plot_learning_curve(history):
    loss = history.history['loss']
    epochs = [i for i, _ in enumerate(loss)]
    plt.scatter(epochs, loss, color='skyblue')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.show()


def random_predict(prediction, temperature):
    prediction = np.asarray(prediction).astype('float64')
    log_pred = np.log(prediction) / temperature
    exp_pred = np.exp(log_pred)
    final_pred = exp_pred / np.sum(exp_pred)
    random_pred = np.random.multinomial(1, final_pred)
    return random_pred


def generate_word(model, data, num, seed, char_dict, temperature=1, max_len=60):
    entire_text = list(data[seed])
    prediction = random_predict(model.predict([[entire_text[num: num + max_len]]])[0], temperature)
    entire_text.append(prediction)
    reverse_char_dict = {value: key for key, value in char_dict.items()}
    generated_text = ''
    for char_vec in entire_text:
        index = np.argmax(char_vec)
        generated_text += reverse_char_dict[index]
    return generated_text


def generate_text(model, data, iter_num, seed, char_dict, temperature=1, max_len=60):
    entire_text = list(data[seed])
    for i in range(iter_num):
        prediction = random_predict(model.predict([[entire_text[i: i + max_len]]])[0], temperature)
        entire_text.append(prediction)
    reverse_char_dict = {value: key for key, value in char_dict.items()}
    generated_text = ''
    for char_vec in entire_text:
        index = np.argmax(char_vec)
        generated_text += reverse_char_dict[index]
    return generated_text


def vary_temperature(temp_lst, model, data, iter_num, seed, char_dict):
    for temperature in temp_lst:
        print("Generated text at temperature {0}:\n{1}\n\n".format(temperature, generate_text(model, data, iter_num, seed, char_dict, temperature)))


if __name__ == '__main__':
    text_data = get_text()
    print("Character length: {0}".format(len(text_data)))  # Character length: 600893
    print(text_data[:100])  # print first 100 characters of text input, just checking

    max_len = 60
    step = 1  # 3
    X, y, char_dict = preprocess_split(text_data, max_len, step)
    vocab_size = len(char_dict)
    print("Number of sequences: {0}\nNumber of unique characters: {1}".format(len(X), vocab_size))

    # model = build_model(max_len, vocab_size)
    # plot_model(model, show_shapes=True, show_layer_names=True) -> not working atm. ToDo for documentation
    # history = model.fit(X, y, epochs=50, batch_size=128)
    # plot_learning_curve(history)

    # model.save('de_3meister_LSTM_model.hdf5')  # -> deprecated
    # model.save('de_3meister_LSTM_model.keras')
    # To load the model:
    model = load_model('model.hdf5')

    # vary_temperature([0.3, 0.6, 0.9, 1.2], model, X, 1000, 10, char_dict)
    temp_lst = [0.3, 0.6, 0.9, 1.2]
    generate_word(model, X, 1000, 10, char_dict)
