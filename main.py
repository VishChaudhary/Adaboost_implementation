import numpy as np
import tensorflow as tf
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from adaboost import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time, resource
from decision_tree import DecisionTreeClassifier_2
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def accuracy_multi_class(confusion_matrix):
    # Calculate accuracy for multi-class classification
    correct = sum(confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    total = sum(sum(row) for row in confusion_matrix)
    return correct / total if total != 0 else 0

def precision_multi_class(confusion_matrix):
    # Calculate precision for each class and return a list of precisions
    precisions = []
    for i in range(len(confusion_matrix)):
        true_positive = confusion_matrix[i][i]
        false_positive = sum(confusion_matrix[row][i] for row in range(len(confusion_matrix))) - true_positive
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        precisions.append(precision)
    return precisions

def recall_multi_class(confusion_matrix):
    # Calculate recall for each class and return a list of recalls
    recalls = []
    for i in range(len(confusion_matrix)):
        true_positive = confusion_matrix[i][i]
        false_negative = sum(confusion_matrix[i]) - true_positive
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        recalls.append(recall)
    return recalls

def f1_score_multi_class(precisions, recalls):
    # Calculate F1-score for each class and return a list of F1-scores
    f1_scores = []
    for i in range(len(precisions)):
        precision = precisions[i]
        recall = recalls[i]
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        f1_scores.append(f1_score)
    return f1_scores

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
    output_shape = np.max(y_train) + 1
    y_train = np.array(y_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape[1],)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15)
    accuracy = model.evaluate(x_train, y_train)[1]
    print(f'Neural Network Accuracy: {accuracy}')
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


def our_decision_tree(x_train, y_train, max_depth = 3):
    model = DecisionTreeClassifier_2()
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    prediction = [val[0] for val in y_pred]
    print(f'Our Decision Tree Accuracy: {accuracy_score(y_pred, y_train)}')
    return prediction, model


def custom_adaboost(x_train,y_train, n_estimators=100, min_samples_split=2):
    x_train = pd.DataFrame(x_train)
    y_train = pd.Series(y_train)

    model = AdaBoostClassifier(n_estimators, min_samples_split)
    model.fit(x_train, y_train)
    prediction = model.predict(x_train)

    # print(f'Custom Stumped Adaboost Accuracy: {accuracy_score(prediction, y_train):}')
    return prediction, model


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
    time_start = time.perf_counter()
    file = 'final_datasets/credit_card_binned.csv'

    num_classifiers = 4
    header, original_data = read_csv_without_libraries(file)
    num_classes = np.max([val[-1] for val in original_data]) + 1
    train_data, x_test, y_test = train_test_split(original_data)

    ada_x_train = [val[:-1] for val in train_data]
    ada_y_train = [val[-1] for val in train_data]

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

        if i == 3:
            predictions, our_decision_tree_model = our_decision_tree(x_train,y_train)
            classifier_models[i] = our_decision_tree_model

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
    #num_classes = np.max(y_test) + 1
    for j in range(0, length_test):
        classifier_prediction = [0] * num_classes
        for i in range(0, num_classifiers):
            classifier_weights[i] = np.log((1-classifier_errors[i])/(classifier_errors[i]))
            if i == 0:
                prediction = np.argmax(classifier_models[i].predict(np.array(x_test[j]).reshape(1,-1),verbose=0))

            elif i == 3:
                prediction = classifier_models[i].predict(np.array(x_test[j]).reshape(1, -1))[0][0]

            else:
                prediction = classifier_models[i].predict(np.array(x_test[j]).reshape(1,-1))[0]

            classifier_prediction[prediction] += classifier_weights[i]

        # print(f'The predicted class is: {np.argmax(classifier_prediction)} and the actual class is: {y_test[j]}')
        predict_array.append(np.argmax(classifier_prediction))

    predict_array = np.array(predict_array)
    y_test = np.array(y_test)
    # print(predict_array)
    # print(y_test)
    incorrect_sum = 0

    for i in range(0, length_test):
        if predict_array[i] != y_test[i]:
            incorrect_sum += 1

    print(f'{incorrect_sum} out of {length_test} values misclassified')

    conf_matrix = confusion_matrix(y_test, predict_array)

    # Extract true positives, true negatives, false positives, and false negatives


    # Calculate metrics
    accuracy = accuracy_multi_class(conf_matrix)
    precision = precision_multi_class(conf_matrix)
    recall = recall_multi_class(conf_matrix)
    f1 = f1_score_multi_class(precision, recall)

    # Print the results
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precisions: {sum(precision)/len(precision):.2f}")
    print(f"Recalls: {sum(recall)/len(recall):.2f}")
    print(f"F1 Scores: {sum(f1)/len(f1):.2f}")

    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print(f"Adaboost without stumping took %5.4f secs %5.4f MByte" % (time_elapsed, memMb))
    print(f'Stumped Adaboost implementation:\n')
    time_start = time.perf_counter()
    _, ada_model = custom_adaboost(ada_x_train,ada_y_train)
    ada_prediction = ada_model.predict(pd.DataFrame(x_test))
    # print(f'Predicted Class: {ada_prediction}')
    incorrect_sum = 0

    for i in range(0, length_test):
        if ada_prediction[i] != y_test[i]:
            incorrect_sum += 1

    print(f'{incorrect_sum} out of {length_test} values misclassified')

    conf_matrix = confusion_matrix(y_test, ada_prediction)

    # Extract true positives, true negatives, false positives, and false negatives
    accuracy = accuracy_multi_class(conf_matrix)
    precision = precision_multi_class(conf_matrix)
    recall = recall_multi_class(conf_matrix)
    f1 = f1_score_multi_class(precision, recall)

    # Print the results
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precisions: {sum(precision)/len(precision):.2f}")
    print(f"Recalls: {sum(recall)/len(recall):.2f}")
    print(f"F1 Scores: {sum(f1)/len(f1):.2f}")
    time_elapsed = (time.perf_counter() - time_start)
    memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    print(f"Adaboost with stumping took %5.4f secs %5.4f MByte" % (time_elapsed, memMb))


    

if __name__ == "__main__":
    main()
