import csv
import random
import math


def main():
    # Read the file into (data) as a list of dictionaries.
    with open('Iris.csv') as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=','))

    # Shuffle every specie separated then join them in one list.
    shuffled_data = random.sample(data[0:50], 50) + random.sample(data[50:100], 50) + random.sample(data[100:150], 50)

    # Select training data and validation data (5-Folds, 80%-20%)
    t_data = shuffled_data[0:40] + shuffled_data[50:90] + shuffled_data[100:140]
    v_data = shuffled_data[40:50] + shuffled_data[90:100] + shuffled_data[140:150]

    NN = None
    K = 1
    A1 = math.inf
    A2 = math.inf

    print("Test datum number | Actual Specie | Nearest Neighbor number | Expected specie")
    for v_datum in v_data:
        A1 = math.inf
        A2 = math.inf
        NN = t_data[0]  # initialization
        for t_datum in t_data:
            dist = abs(float(t_datum['SepalLengthCm'])-float(v_datum['SepalLengthCm'])) +\
                 abs(float(t_datum['SepalWidthCm']) - float(v_datum['SepalWidthCm'])) + \
                 abs(float(t_datum['PetalLengthCm']) - float(v_datum['PetalLengthCm'])) + \
                 abs(float(t_datum['PetalWidthCm']) - float(v_datum['PetalWidthCm']))
            if dist < A1:
                A1 = dist
                NN = t_datum
        print(v_datum['Id'] + "\t\t  | " + v_datum['Species']
              + "\t\t  | " + NN['Id'] + "\t\t\t| " + NN['Species'])


if __name__ == "__main__":
    main()

