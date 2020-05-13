import skfuzzy as fuzz
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from random import randint
from GeneticAlgorithm import *


def parse_data(data, terms, linguistic):
    # data.insert(2, "Age", [21, 23, 24, 21, 23, 24, 21], True)

    attributes = list(linguistic.keys())
    col_index = 0
    for i in range(len(attributes)):
        attribute = list(data[attributes[i]])
        x_max = max(attribute) + 1
        x_axis = np.arange(0, x_max, 1)
        term_set = terms[i]
        attribute_index = 0
        for term_value in term_set:
            attribute_value = linguistic.get(attributes[i])[attribute_index]
            new_values = list()
            for attr_value in attribute:
                rez = fuzz.interp_membership(x_axis, term_value, attr_value)
                new_values.append(rez)
            col_name = attributes[i] + '_' + attribute_value
            data.insert(col_index, col_name, new_values, True)
            col_index += 1
            attribute_index += 1
        data.drop(attributes[i], axis='columns', inplace=True)
    print('Data has been parsed.')
    return data


def get_terms(individual):
    terms_individual = []
    for attribute in individual:
        terms_num = len(attribute)
        terms = []
        for i in range(terms_num):
            numbers = np.arange(0, attribute[i][1] + 1, 1)
            if 0 < i < terms_num - 1:
                term_numbers = attribute[i][0]
                term_numbers.sort()
                term = fuzz.trapmf(numbers, term_numbers)
            elif i == 0:
                term = fuzz.zmf(numbers, attribute[i][0][0], attribute[i][1])
            else:
                term = fuzz.smf(numbers, attribute[i][0][0], attribute[i][1])
            terms.append(term)
        terms_individual.append(terms)
    return terms_individual


def get_common_p_and_entropy(data):
    target_attr = data.iloc[:, -1]
    p_yes = 0
    p_no = 0
    for i in range(len(target_attr)):
        p_yes += target_attr[i]
        p_no += (1.0 - target_attr[i])
    p = p_yes + p_no
    e = -(p_yes / p) * math.log2(p_yes / p) - (p_no / p) * math.log2(p_no / p)
    return p, e


def entropy(data, attribute):
    target_attr = data.iloc[:, -1]
    p_yes = 0
    p_no = 0
    p_common = get_common_p_and_entropy(data)[0]
    e = 0

    attr_frame = data.loc[:, data.columns.to_series()
                                 .str.contains(attribute).tolist()]
    for attr_value in attr_frame:
        value_sum = data[attr_value].sum()
        for i in range(len(target_attr)):
            p_yes += min(target_attr[i], data[attr_value][i])
            p_no += min((1.0 - target_attr[i]), data[attr_value][i])
        p = p_yes + p_no
        e_attr = -(p_yes / p) * math.log2(p_yes / p) - (p_no / p) * math.log2(p_no / p)
        e += (value_sum / p_common) * e_attr
    return e


def get_gain(data, attribute):
    gain = get_common_p_and_entropy(data)[0] - entropy(data, attribute)
    return gain


def get_priority(data, linguistic):
    attributes = list(linguistic.keys())
    attr_len = len(attributes)
    for i in range(attr_len):
        for j in range(attr_len - i - 1):
            if get_gain(data, attributes[j]) < get_gain(data, attributes[j + 1]):
                attributes[j], attributes[j + 1] = attributes[j + 1], attributes[j]
    return attributes


def get_tree_data(data, attribute_values, priority):
    columns = pd.MultiIndex.from_product(attribute_values)
    new_data = []
    rows_num = data.shape[0]
    for i in range(rows_num):
        row = data.iloc[i, :]
        new_row = []
        for av in columns:
            new_value_arr = []
            for j in range(len(av)):
                name = priority[j] + '_' + av[j]
                v = row[name]
                new_value_arr.append(v)
            new_value = min(new_value_arr)
            new_row.append(new_value)
        new_data.append(new_row)

    new_df = pd.DataFrame(new_data, columns=columns)
    return new_df


def get_coefficients_data(data, p_array, positive):
    coefficients_data = data.copy(deep=True)
    for cd in coefficients_data:
        for i in range(len(p_array)):
            if positive:
                p = min(p_array[i], coefficients_data[cd][i])
            else:
                p = min((1.0 - p_array[i]), coefficients_data[cd][i])
            coefficients_data[cd][i] = p
    return coefficients_data


def generate_tree(data, term_numbers, terms):
    linguistic = get_linguistic(data, term_numbers)
    parsed_data = parse_data(data, terms, linguistic)
    priority = get_priority(data, linguistic)

    attribute_values = []
    for p in priority:
        attribute_values.append(linguistic.get(p))
    tree_data = get_tree_data(data, attribute_values, priority)
    tree_data_copy = tree_data.copy(deep=True)
    positive_coefficients_data = get_coefficients_data(tree_data_copy,
                                                       parsed_data.iloc[:, -1], True)
    negative_coefficients_data = get_coefficients_data(tree_data_copy,
                                                       parsed_data.iloc[:, -1], False)

    return [tree_data, positive_coefficients_data, negative_coefficients_data, parsed_data.iloc[:, -1]]


def find_the_best_tree():
    data = pd.read_csv('data.csv', sep=',')
    population = generate_population(data, 20, 3)
    term_numbers = get_numbers_of_terms(data)

    generation = 0
    accuracy_max = 0
    the_best_tree = []
    while generation < 20:
        accuracies = []
        for p in population:
            data_copy = pd.read_csv('data.csv', sep=',')
            terms = get_terms(p)
            tree_data = generate_tree(data_copy, term_numbers, terms)
            accuracy = get_accuracy(tree_data)
            if accuracy >= 0.95:
                return tree_data[0]
            if accuracy > accuracy_max:
                accuracy_max = accuracy
                the_best_tree.clear()
                the_best_tree.append(tree_data[0])
            accuracies.append(accuracy)
        population_priority = get_random_couples(accuracies, 20)
        population = get_new_generation(population, population_priority)
        generation += 1
        print('Generation ' + str(generation))
    return the_best_tree[0]


def main():
    the_best_tree = find_the_best_tree()
    return the_best_tree


if __name__ == '__main__':
        main()
