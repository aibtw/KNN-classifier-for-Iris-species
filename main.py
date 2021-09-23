import csv
import random
import math
from statistics import mode


def main():
    # Reading the input CSV file.
    with open('Iris.csv') as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=','))

    # Change numeric data from string to float, to make calculations possible.
    for dictionary in data:
        for col in dictionary:
            if col != 'Species' and col != 'Id': dictionary[col] = float(dictionary[col])

    # TODO: make the following lines adaptive to any number of inputs, not necessarily 150.
    # TODO: Fix the seed of random to one value to have constant shuffling each time the code run.
    # Shuffle the data, keeping species separated.
    data = random.sample(data[0:50], 50) + random.sample(data[50:100], 50) + random.sample(data[100:150], 50)

    # Select training data (t_data) and validation data (v_data) (5-Folds, 80%-20%), and value of K
    t_data = data[0:40] + data[50:90] + data[100:140]
    v_data = data[40:50] + data[90:100] + data[140:150]
    K = 2

    expected_label = []

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

            # Add each train datum to temp arrays as a tuple (Train datum, its distance from current validation data)
            A1_temp.append((t_datum, A1_dist))
            A2_temp.append((t_datum, A2_dist))

        # Sort the training data by their distance from the current validation datum
        A1_temp = sorted(A1_temp, key=lambda k: k[1])
        nearest_neighbours = A1_temp[:K]

        nns_labels = []
        for neighbour in nearest_neighbours:
            nns_labels.append(neighbour[0]['Species'])

        expected_label.append(mode(nns_labels))

    # TODO improve output form
    for i in range(len(v_data)):
        print("Validation data ID: ", v_data[i]['Id'],
              " | Expected label: ", expected_label[i],
              " | Actual label: ", v_data[i]['Species'])


if __name__ == "__main__":
    main()
