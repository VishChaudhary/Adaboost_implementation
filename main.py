import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier

# Read in data
def read_csv_without_libraries(filename, delimiter=","):
    with open(filename, "r") as f:
        headers = f.readline().strip().split(delimiter)
        data = []
        for line in f:
            row = [(int(float(entry.strip()))) for entry in line.split(delimiter)]
            data.append(row)

    return headers, data


# Initial weight of 1/d to all
# mod_data -> List[Tuple(List[],List[])]
# Tuple: [i:0-len(data)][0][0]
# Data:  [i:0-len(data][1][k:len(row) {this is for accessing and changing a row in the data}]
def initial_weights(data):
    weight = 1 / len(data)

    mod_data = [([weight], row) for row in data]
    return mod_data


def resample(data, num_sample=None):
    num_sample = num_sample or len(data)

    # Get weight for each training example and append into 1 list
    offset = 1e-9
    weights = []

    for row in data:
        if row[0][0] == 0:
            weights.append(offset)
        else:
            weights.append(row[0][0])

    # Sample indices with replacement with the weight of each training set being its probability of being picked
    indices = np.random.choice(len(data), size=num_sample, replace=True, p=weights / np.sum(weights))

    data_resampled = []
    for i in indices:
        data_resampled.append(data[i])
    return data_resampled, weights


def generate_training_sets(data):
    x_train = []
    y_train = []
    for element in data:
        x_train.append(element[1][:-1])
        y_train.append(element[1][-1])
    return x_train, y_train


def train_test_split(data, split = 0.1):
    np.random.shuffle(data)
    length = len(data)
    split_index = int(length*split)
    test_data = data[:split_index]
    train_data = data[split_index:]
    x_test = []
    y_test = []
    for row in test_data:
        x_test.append(row[:-1])
        y_test.append(row[-1])
    return train_data, x_test, y_test


def MLP(x_train, y_train):
    x_train = np.array(x_train)
    input_shape = x_train.shape
    output_shape = len(set(y_train)) + 1
    y_train = np.array(y_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    accuracy = model.evaluate(x_train, y_train)[1]
    print(f'Accuracy: {accuracy}')
    predictions = [np.argmax(prob) for prob in model.predict(x_train,verbose=0)]
    return predictions, model


def svm_classifier(x_train, y_train):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model = svm.SVC()
    model.fit(x_train,y_train)
    accuracy = model.score(x_train,y_train)
    predictions = model.predict(x_train)
    print(f'Svm accuracy: {accuracy}')
    return predictions, model


def decision_tree(x_train,y_train,criterion = 'gini',max_depth=None):
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)
    accuracy = model.score(x_train,y_train)
    print(f'Sklearn Decision Tree Accuracy: {accuracy}')
    predictions = model.predict(x_train)
    return predictions,model


def was_missclass(predictions, y_train, x_train, weighted_original_data, weights):
    is_misclassified = [False] * len(predictions)
    error = 0.0
    for i, predict_val in enumerate(predictions):
        if y_train[i] != predict_val:
            index = return_index(weighted_original_data, x_train[i])
            error += weights[index]
            is_misclassified[i] = True
    return is_misclassified, error


def return_index(sampled_data, x_train):
    for index, val in enumerate(sampled_data):
        if x_train == val[1][:-1]:
            return index
    print('Fail')
    return 0


def update_weights(was_misclassified, error, weights, weighted_original_data, x_train):
    bias = error / (1 - error)
    old_weights_sum = np.sum(weights)
    for i, val in enumerate(was_misclassified):
        if not val:
            index = return_index(weighted_original_data, x_train[i])
            weighted_original_data[i][0][0] *= bias
            weights[index] *= bias

    new_weights_sum = np.sum(weights)
    norm_term = old_weights_sum / new_weights_sum
    new_weights = []
    for row in weighted_original_data:
        row[0][0] *= norm_term
        new_weights.append(row[0][0])
    return new_weights, weighted_original_data


def main():
    file = 'wine_binned.csv'

    num_classifiers = 3
    header, original_data = read_csv_without_libraries(file)

    train_data, x_test, y_test = train_test_split(original_data)

    # Add initial weight to data
    weighted_train_data = initial_weights(train_data)

    classifier_errors = [-1]*num_classifiers
    classifier_models = [0]*num_classifiers

    for i in range(0, num_classifiers):
        resampled_data, weights = resample(weighted_train_data)
        x_train, y_train = generate_training_sets(resampled_data)
        if i == 0:
            predictions, mlp_model = MLP(x_train,y_train)
            classifier_models[i] = mlp_model
        if i == 1:
            predictions, svm_model = svm_classifier(x_train, y_train)
            classifier_models[i] = svm_model
        if i == 2:
            predictions, decision_tree_model = decision_tree(x_train,y_train)
            classifier_models[i] = decision_tree_model

        was_misclassified, error = was_missclass(predictions, y_train, weights=weights, x_train=x_train,
                                                 weighted_original_data=weighted_train_data)
        if error == 0:
            error = 1e-6
        classifier_errors[i] = error
        weights, weighted_train_data = update_weights(was_misclassified, error, weights, weighted_train_data,
                                                         x_train)

    classifier_weights = [0]*num_classifiers
    length_test = len(x_test)
    predict_array = []

    for j in range(0, length_test):
        classifier_prediction = [0] * (num_classifiers + 1)
        for i in range(0,num_classifiers):
            classifier_weights[i] = np.log((1-classifier_errors[i])/(classifier_errors[i]))
            if i == 0:
                prediction = np.argmax(classifier_models[i].predict(np.array(x_test[j]).reshape(1,-1),verbose=0))

            else:
                prediction = int(classifier_models[i].predict(np.array(x_test[j]).reshape(1,-1))[0])

            classifier_prediction[prediction] += classifier_weights[i]

        print(f'The predicted class is: {np.argmax(classifier_prediction)} and the actual class is: {y_test[j]}')
        predict_array.append(np.argmax(classifier_prediction))

    predict_array = np.array(predict_array)
    y_test = np.array(y_test)
    print(predict_array)
    print(y_test)
    incorrect_sum = 0

    for i in range(0, length_test):
        if predict_array[i] != y_test[i]:
            incorrect_sum += 1

    print(f'{incorrect_sum} out of {length_test} values misclassified')


if __name__ == "__main__":
    main()
