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
            row = [float(entry.strip()) for entry in line.split(delimiter)]
            data.append(row)

    return headers, data


# Initial weight of 1/d to all
def initial_weights(data):
    weight = 1 / len(data)
    # mod_data -> List[Tuple(List[],List[])]
    #Tuple: [i:0-len(data)][0][0]
    #Data:  [i:0-len(data][1][k:len(row) {this is for accessing and changing a row in the data}]
    mod_data = [([weight], row) for row in data]
    return mod_data


def resample(data, num_sample=None):
    num_sample = num_sample or len(data)
    # Get weight for each training example and append into 1 list
    offset = 1e-9
    weights = []
    # weights = [row[0][0] for row in data]
    # weights = [offset for val in weights if val == 0]

    for row in data:
        if row[0][0] == 0:
            weights.append(offset)
        else:
            weights.append(row[0][0])

    #print(weights)
    # print(f'Hellooooo {prob[0:5]}')
    # Sample indices with replacement with the weight of each training set being its probability of being picked
    indices = np.random.choice(len(data), size=num_sample, replace=True, p=weights / np.sum(weights))
    # print(indices)
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


def MLP(x_train, y_train):
    x_train = np.array(x_train)
    input_shape = x_train.shape
    output_shape = len(set(y_train)) + 1
    # print(f' input_shape: {input_shape}')
    # print(f' input_shape[1]: {input_shape[1]}')
    y_train = np.array(y_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(input_shape[1],)),
        #tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)
    accuracy = model.evaluate(x_train, y_train)[1]
    print(f'Accuracy: {accuracy}')
    # predictions = np.array([np.argmax(prob) for prob in model.predict(x_train)])
    predictions = [np.argmax(prob) for prob in model.predict(x_train)]
    return predictions, model
    # print(y_train)

def svm_classifier(x_train, y_train):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    model = svm.SVC()
    model.fit(x_train,y_train)
    accuracy = model.score(x_train,y_train)
    predictions = model.predict(x_train)
    #predictions = int(predictions)
    print(f'Svm accuracy: {accuracy}')
    # print(f'Svm predictions:\n{predictions}')
    return predictions, model

def decision_tree(x_train,y_train,criterion = 'gini',max_depth=None):
    model = DecisionTreeClassifier()
    model.fit(x_train,y_train)
    accuracy = model.score(x_train,y_train)
    print(f'Sklearn Decision Tree Accuracy: {accuracy}')
    predictions = model.predict(x_train)
    #print(f'Sklearn Decision Tree Predictions:\n{predictions}')
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
    #length = len(was_misclassified)
    old_weights_sum = np.sum(weights)
    for i, val in enumerate(was_misclassified):
        if not val:
            # for j in range(0,length):
            #     if x_train[i] == sampled_data[j][1]:
            #         sampled_data[j][0][0] *= bias
            #         new_weights_sum += sampled_data[j][0][0]
            #         print(f'New weights sum: {new_weights_sum}')
            index = return_index(weighted_original_data, x_train[i])
            weighted_original_data[i][0][0] *= bias
            weights[index] *= bias
            #new_weights_sum += weights[index]
            # print(f'New weights sum: {new_weights_sum}')
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
    seudo_test = original_data[:10]
    x_test = []
    y_test = []
    for i in seudo_test:
        x_test.append(i[:-1])
        y_test.append(i[-1])

    # for i in range(0, 5):
    #     print(data[i])

    # Add initial weight to data
    weighted_original_data = initial_weights(original_data)
    #
    # for i in range(0,2):
    #     data[i][0][0] = 1.5
    #
    # for i in range(0,15):
    #     print(data[i])
############################
    # # Sample data
    # resampled_data, weights = resample(weighted_original_data)
    # # for i in range(0, 5):
    # #     print(sampled_data[i])
    #
    # x_train, y_train = generate_training_sets(resampled_data)
    # predictions, mlp_model = MLP(x_train, y_train)
    #
    # was_misclassified, error = was_missclass(predictions, y_train, weights=weights, x_train=x_train,
    #                                          weighted_original_data=weighted_original_data)
    # # print(f'shape: {np.shape(np.array(x_train[0]))}')
    # # predic = mlp_model.predict(np.array(x_train[:5]))
    # # print(predic)
    # # print(y_train[:5])
    # weights, weighted_original_data = update_weights(was_misclassified, error, weights, weighted_original_data, x_train)
    #
    # resampled_data, weights = resample(weighted_original_data)
    # x_train, y_train = generate_training_sets(resampled_data)
    # predictions, model = svm_classifier(x_train, y_train)
    # was_misclassified, error = was_missclass(predictions, y_train, weights=weights, x_train=x_train,
    #                                          weighted_original_data=weighted_original_data)
    # weights, weighted_original_data = update_weights(was_misclassified, error, weights, weighted_original_data, x_train)
    # resampled_data, weights = resample(weighted_original_data)
    # x_train, y_train = generate_training_sets(resampled_data)
    # predictions, model = decision_tree(x_train,y_train)
################################
    classifier_errors = [-1]*(num_classifiers)
    classifier_models = [0]*(num_classifiers)
    #classifiers = [MLP(x_train,y_train),svm_classifier(x_train,y_train),decision_tree(x_train,y_train)]
    for i in range(0, num_classifiers):
        resampled_data, weights = resample(weighted_original_data)
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
                                                 weighted_original_data=weighted_original_data)
        classifier_errors[i] = error
        weights, weighted_original_data = update_weights(was_misclassified, error, weights, weighted_original_data,
                                                         x_train)

    # svm_predictor = classifier_models[1]
    # pred = svm_predictor.predict(x_test)
    # print(pred)
    # print(y_test)

    # for val in weighted_original_data:
    #     print(val)
    # print(np.array(weights))

    # for i in range(0,5):
    #     print(x_train[i])
    #     print(y_train[i])


if __name__ == "__main__":
    main()
