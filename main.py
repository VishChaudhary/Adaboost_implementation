import numpy as np
import tensorflow as tf

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
    # mod_data -> List(Tuple(List,List))
    mod_data = [([weight], row) for row in data]
    return mod_data


def resample(data, num_sample = None):
    num_sample = num_sample or len(data)
    # Get weight for each training example and append into 1 list
    prob = [row[0][0] for row in data]
    #print(f'Hellooooo {prob[0:5]}')
    # Sample indices with replacement with the weight of each training set being its probability of being picked
    indices = np.random.choice(len(data), size=num_sample, replace=True, p=prob/np.sum(prob))
    #print(indices)
    data_resampled = []
    for i in indices:
        data_resampled.append(data[i])
    return data_resampled, prob


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

    y_train = np.array(y_train)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train,y_train,epochs=5)
    accuracy = model.evaluate(x_train,y_train)[1]
    print(f'Accuracy: {accuracy}')
    #predictions = np.array([np.argmax(prob) for prob in model.predict(x_train)])
    predictions = [np.argmax(prob) for prob in model.predict(x_train)]
    return predictions, accuracy
    # print(y_train)

def match(sampled_data, x_train):
    for i in range(0, len(x_train)):
        for index, element in enumerate(sampled_data):
            if x_train[i] == element[1][:-1]:
                return index
    print('Fail')
    return 0

def update_weights(was_misclassified,error,weights,sampled_data,x_train):
    bias = error/(1-error)
    length = len(predictions)
    old_weights_sum = np.sum(weights)
    new_weights_sum = 0.0
    for i in range(0,length):
        if not was_misclassified[i]:
            # for j in range(0,length):
            #     if x_train[i] == sampled_data[j][1]:
            #         sampled_data[j][0][0] *= bias
            #         new_weights_sum += sampled_data[j][0][0]
            #         print(f'New weights sum: {new_weights_sum}')
            j = match(sampled_data, x_train)
            sampled_data[j][0][0] *= bias
            new_weights_sum += sampled_data[j][0][0]
            #print(f'New weights sum: {new_weights_sum}')

    norm_term = old_weights_sum/new_weights_sum
    for row in sampled_data:
        row[0][0] *= norm_term
    new_weights = [row[0][0] for row in sampled_data]
    return new_weights, sampled_data

file = 'wine_binned.csv'

header, data = read_csv_without_libraries(file)
# for i in range(0, 5):
#     print(data[i])

# Add initial weight to data
data = initial_weights(data)
#
# for i in range(0,2):
#     data[i][0][0] = 1.5
#
# for i in range(0,15):
#     print(data[i])

# Sample data
sampled_data, weights = resample(data)
# for i in range(0, 5):
#     print(sampled_data[i])

x_train, y_train = generate_training_sets(sampled_data)
predictions, accuracy = MLP(x_train, y_train)

error = 0.0
was_misclassified = [False]*len(predictions)
for i in range(0,len(predictions)):
    if y_train[i]==predictions[i]:
        error += weights[i]
        was_misclassified[i]=True

weights, sampled_data = update_weights(was_misclassified,error,weights,sampled_data,x_train)
for val in sampled_data:
    print(val)
print(np.array(weights))


# for i in range(0,5):
#     print(x_train[i])
#     print(y_train[i])
# if __name__ = "__main":
#


