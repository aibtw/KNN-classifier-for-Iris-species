"""
22/9/2021
EE482
Ahmad Batwa - 1845044
"""
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
    random.seed(1000000)
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

    avg_accuracies = {'K': [], 'L1': [], 'L2': []}

    # Repeat the test for different values of K
    for K in range(1, 61, 2):
        all_acc = np.zeros([5, 2])
        # For each value of K, repeat the test with a different combination of data sets
        for i in range(5):
            # KNN function returns a list [accuracy of L1, accuracy of L2]
            all_acc[i] = knn(K, t_data_sets[i], v_data_sets[i])

        avg_accuracies['K'].append(K)
        avg_accuracies['L1'].append(mean(all_acc[:, 0]))
        avg_accuracies['L2'].append(mean(all_acc[:, 1]))

    # Write to output file
    with open("output.csv", "w", newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['K', 'L1', 'L2'])  # The header
        for i in range(len(avg_accuracies['K'])):
            writer.writerow([avg_accuracies['K'][i], avg_accuracies['L1'][i], avg_accuracies['L2'][i]])


def knn(K, t_data, v_data):
    """
    A function that receives training data, validation data, and number of neighbours (K), then implements KNN algorithm
    using both L1 and L2 distances, and returns the accuracy of the algorithm

    :param K: Number of nearest neighbours to consider
    :param t_data: Training data
    :param v_data: Validation data
    :return: a list [Algorithms' accuracy for L1 distance, Algorithms' accuracy for L2 distance]
    """

    expected_labels_L1 = []
    expected_labels_L2 = []

    for v_datum in v_data:
        L1_temp = []
        L2_temp = []
        for t_datum in t_data:
            L1_dist = abs(t_datum['SepalLengthCm'] - v_datum['SepalLengthCm']) + \
                      abs(t_datum['SepalWidthCm'] - v_datum['SepalWidthCm']) + \
                      abs(t_datum['PetalLengthCm'] - v_datum['PetalLengthCm']) + \
                      abs(t_datum['PetalWidthCm'] - v_datum['PetalWidthCm'])

            L2_dist = (t_datum['SepalLengthCm'] - v_datum['SepalLengthCm'])**2 + \
                      (t_datum['SepalWidthCm'] - v_datum['SepalWidthCm'])**2 + \
                      (t_datum['PetalLengthCm'] - v_datum['PetalLengthCm'])**2 + \
                      (t_datum['PetalWidthCm'] - v_datum['PetalWidthCm'])**2

            # Add each train datum to temp array as tuple(Train datum, its distance from current validation datum)
            L1_temp.append((t_datum, L1_dist))
            L2_temp.append((t_datum, L2_dist))

        # Sort the training data by their distance from the current validation datum
        L1_temp = sorted(L1_temp, key=lambda k: k[1])
        L2_temp = sorted(L2_temp, key=lambda k: k[1])

        nearest_neighbours_L1 = L1_temp[:K]
        nearest_neighbours_L2 = L2_temp[:K]

        nns_labels_L1 = []
        nns_labels_L2 = []
        for neighbour in nearest_neighbours_L1:
            nns_labels_L1.append(neighbour[0]['Species'])
        for neighbour in nearest_neighbours_L2:
            nns_labels_L2.append(neighbour[0]['Species'])

        expected_labels_L1.append(mode(nns_labels_L1))
        expected_labels_L2.append(mode(nns_labels_L2))

    true_L1 = 0
    true_L2 = 0

    # L1
    for i in range(len(v_data)):
        #                   Debug area
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_L1[i],
        #       " | Actual label: ", v_data[i]['Species'])

        if expected_labels_L1[i] == v_data[i]['Species']:
            true_L1 += 1

    # L2
    for i in range(len(v_data)):
        #                   Debug area
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_L2[i],
        #       " | Actual label: ", v_data[i]['Species'])

        if expected_labels_L2[i] == v_data[i]['Species']:
            true_L2 += 1

    accuracy_L1 = true_L1/30*100
    accuracy_L2 = true_L2/30*100

    return [accuracy_L1, accuracy_L2]


if __name__ == "__main__":
    main()
