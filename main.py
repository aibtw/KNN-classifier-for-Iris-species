import csv
import random
from statistics import mode


def main():
    # Reading the input CSV file.
    with open('Iris.csv') as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=','))

    # Change numeric data from string to float, to make calculations possible.
    for dictionary in data:
        for col in dictionary:
            if col != 'Species' and col != 'Id': dictionary[col] = float(dictionary[col])

    # Shuffle the data, keeping species separated.
    random.seed(100)  # Fixed to 100
    data = random.sample(data[0:50], 50) + random.sample(data[50:100], 50) + random.sample(data[100:150], 50)

    # Select training data (t_data) and validation data (v_data) (5-Folds, 80%-20%), and value of K
    t_data = data[0:40] + data[50:90] + data[100:140]
    v_data = data[40:50] + data[90:100] + data[140:150]

    # Select train data and validation data as 5-Folds(80%-20%)
    # Also, repeat the experiment 5 times, use different set each time

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

    accuracies = []

    for i in range(5):
        # Repeat the test for 65 different values of K
        for K in range(1,130,2):
            # print("============ K = ", K, " | Set Number: ", i+1, " ============")
            accuracies.append(KNN(K, t_data, v_data))


def KNN(K, t_data, v_data):

    expected_labels_A1 = []
    expected_labels_A2 = []

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

    # TODO improve output form
    # print('---------------- A1 ----------------')
    True_A1 = 0
    True_A2 = 0
    for i in range(len(v_data)):
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_A1[i],
        #       " | Actual label: ", v_data[i]['Species'])
        if expected_labels_A1[i] == v_data[i]['Species']: True_A1 += 1
    # print('---------------- A2 ----------------')
    for i in range(len(v_data)):
        # print("Validation data ID: ", v_data[i]['Id'],
        #       " | Expected label: ", expected_labels_A2[i],
        #       " | Actual label: ", v_data[i]['Species'])
        if expected_labels_A2[i] == v_data[i]['Species']: True_A2 += 1
    accuracy_A1 = True_A1/30*100
    accuracy_A2 = True_A2/30*100
    # print('Accuracy in A1: ', accuracy_A1, '\nAccuracy in A2: ', accuracy_A2)
    return [accuracy_A1, accuracy_A2]
    # Accuracy = (sum of true)/(total)
    # /
    # ========================|========================= Prediction ====================================================
    #   Ground truth          |-----------------------------------------------------------------------------------------
    # ========================| Class 1     | Class 2       | Class 3 |
    # ------------------------------------------------------------------------------------------------------------------
    #   Class 1               | True C1     | False C2      | False C3|
    # ------------------------------------------------------------------------------------------------------------------
    #   Class 2               | False C1    | True C2       | False C3|
    # ------------------------------------------------------------------------------------------------------------------
    #   Class 3               | False C1    | False C2      | True C3 |
    # /#


if __name__ == "__main__":
    main()
