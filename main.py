import csv
import random
from statistics import mode, mean
import numpy as np


def main():
    """
    A program that reads a CSV file and applies KNN algorithm on it, then outputs the accuracy of the algorithm in CSV.
    The algorithm is repeated for wide range of K values and 5 different combinations of train-validation data.
    The output CSV file shows the (average accuracy of the 5 combinations) for each K value.
    """
    # Reading input
    with open('Iris.csv') as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=','))

    # Change numeric data from string to float
    for dictionary in data:
        for col in dictionary:
            if col != 'Species' and col != 'Id': dictionary[col] = float(dictionary[col])

    # Shuffle the data of each specie
    random.seed(10406)
    data = random.sample(data[0:50], 50) + random.sample(data[50:100], 50) + random.sample(data[100:150], 50)

    # Train data and validation data are divided as 5-Folds(80%-20%)
    # 5 different sets will be prepared and used iteratively, then the average of their result will be taken

    # 5 sets for train data
    t_data_sets = [data[0:40] + data[50:90] + data[100:140],
                   data[0:30] + data[40:80] + data[90:130] + data[140:150],
                   data[0:20] + data[30:70] + data[80:120] + data[130:150],
                   data[0:10] + data[20:60] + data[70:110] + data[120:150],
                   data[10:50] + data[60:100] + data[110:150]]
    # 5 sets of validation data
    v_data_sets = [data[40:50] + data[90:100] + data[140:150],
                   data[30:40] + data[80:90] + data[130:140],
                   data[20:30] + data[70:80] + data[120:130],
                   data[10:20] + data[60:70] + data[110:120],
                   data[0:10] + data[50:60] + data[100:110]]

    avg_accuracies = {'K': [], 'A1': [], 'A2': []}

    # Repeat the test for 65 different values of K
    for K in range(1, 130, 2):
        all_acc = np.zeros([5, 2])
        # For each value of K, repeat the test with a different combination of data sets
        for i in range(5):
            # KNN function returns a list [accuracy of A1, accuracy of A2]
            all_acc[i] = knn(K, t_data_sets[i], v_data_sets[i])

        avg_accuracies['K'].append(K)
        avg_accuracies['A1'].append(mean(all_acc[:, 0]))
        avg_accuracies['A2'].append(mean(all_acc[:, 1]))

    # Write to output file
    with open("output.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['K', 'A1', 'A2'])  # The header
        for i in range(len(avg_accuracies['K'])):
            writer.writerow([avg_accuracies['K'][i], avg_accuracies['A1'][i], avg_accuracies['A2'][i]])


def knn(K, t_data, v_data):
    """
    A function that receives training data, validation data, and number of neighbours (K), then implements KNN algorithm
    using both A1 and A2 distances, and returns the accuracy of the algorithm

    :param K: Number of nearest neighbours to consider
    :param t_data: Training data
    :param v_data: Validation data
    :return: a list [Algorithms' accuracy for A1 distance, Algorithms' accuracy for A2 distance]
    """

    expected_labels_A1 = []
    expected_labels_A2 = []

    # TODO: Consider using numpy
    for v_datum in v_data:
        A1_temp = []
        A2_temp = []
        for t_datum in t_data:
            A1_dist = abs(t_datum['SepalLengthCm'] - v_datum['SepalLengthCm']) + \
                      abs(t_datum['SepalWidthCm'] - v_datum['SepalWidthCm']) + \
                      abs(t_datum['PetalLengthCm'] - v_datum['PetalLengthCm']) + \
                      abs(t_datum['PetalWidthCm'] - v_datum['PetalWidthCm'])

            A2_dist = (t_datum['SepalLengthCm'] - v_datum['SepalLengthCm'])**2 + \
                      (t_datum['SepalWidthCm'] - v_datum['SepalWidthCm'])**2 + \
                      (t_datum['PetalLengthCm'] - v_datum['PetalLengthCm'])**2 + \
                      (t_datum['PetalWidthCm'] - v_datum['PetalWidthCm'])**2

            # Add each train datum to temp array as tuple(Train datum, its distance from current validation datum)
            A1_temp.append((t_datum, A1_dist))
            A2_temp.append((t_datum, A2_dist))

        # Sort the training data by their distance from the current validation datum
        A1_temp = sorted(A1_temp, key=lambda k: k[1])
        A2_temp = sorted(A2_temp, key=lambda k: k[1])

        nearest_neighbours_A1 = A1_temp[:K]
        nearest_neighbours_A2 = A2_temp[:K]

        nns_labels_A1 = []
        nns_labels_A2 = []
        for neighbour in nearest_neighbours_A1:
            nns_labels_A1.append(neighbour[0]['Species'])
        for neighbour in nearest_neighbours_A2:
            nns_labels_A2.append(neighbour[0]['Species'])

        expected_labels_A1.append(mode(nns_labels_A1))
        expected_labels_A2.append(mode(nns_labels_A2))

    true_A1 = 0
    true_A2 = 0

    # A1
    for i in range(len(v_data)):
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_A1[i],
        #       " | Actual label: ", v_data[i]['Species'])
        if expected_labels_A1[i] == v_data[i]['Species']:
            true_A1 += 1

    # A2
    for i in range(len(v_data)):
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_A2[i],
        #       " | Actual label: ", v_data[i]['Species'])
        if expected_labels_A2[i] == v_data[i]['Species']:
            true_A2 += 1

    accuracy_A1 = true_A1/30*100
    accuracy_A2 = true_A2/30*100

    return [accuracy_A1, accuracy_A2]


if __name__ == "__main__":
    main()
