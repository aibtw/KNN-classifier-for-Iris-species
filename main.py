import csv
import random


def main():
    # Read the file into (data) as a list of dictionaries.
    with open('Iris.csv') as csv_file:
        data = list(csv.DictReader(csv_file, delimiter=','))

    # Shuffle every specie separated then join them in one list.
    shuffled_data = random.sample(data[0:50], 50) + random.sample(data[50:100], 50) + random.sample(data[100:150], 50)

    # Select training data and validation data
    t_data = shuffled_data[0:40] + shuffled_data[50:90] + shuffled_data[100:140]
    v_data = shuffled_data[40:50] + shuffled_data[90:100] + shuffled_data[140:150]

    # algorithm:
    # 1- loop over every V_datum
    # 2- for each v_datum, loop over every t_datum
    # 3- find diff(v_datum, t_datum)
    # 4- if current diff <= previous: current diff is passed to next iteration and previous is dismissed.
    # 5- at the end, get(current_datum.specie) and that is the one.
    # 6- Compare(current_datum.specie, v_datum.specie)
    # 7- Output result.


if __name__ == "__main__":
    main()
