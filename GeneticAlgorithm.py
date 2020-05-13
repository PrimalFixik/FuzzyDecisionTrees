import random
import pandas as pd


# n_terms here is temporarily 3
def get_numbers_of_terms(data):
    numbers = []
    attribute_num = len(data.columns) - 1
    for i in range(attribute_num):
        numbers.append(3)
    return numbers


def generate_population(data, population_length, n_terms):
    seed = 1
    attribute_num = len(data.columns) - 1
    population = []
    for i in range(population_length):
        individual = []
        for j in range(attribute_num):
            attribute = []
            att = data.iloc[:, [j]]
            limit_x = att.max().item()
            term_array = []
            for k in range(n_terms):
                if 0 < k < n_terms - 1:
                    term = get_randoms(seed, limit_x, 4)
                else:
                    term = get_randoms(seed, limit_x, 2)
                seed += 1
                term_array = [term, limit_x]
                attribute.append(term_array)
            individual.append(attribute)
        population.append(individual)
    return population


def get_linguistic(data, term_numbers):
    attribute_num = len(data.columns) - 1
    linguistic = {}
    for i in range(attribute_num):
        name = data.columns[i]
        numbers = []
        for j in range(term_numbers[i]):
            numbers.append('n' + str(j + 1))
        linguistic.update({name: numbers})
    return linguistic


def get_randoms(seed, _max_value=100, _n=10):
    random.seed(seed)
    _result = []
    for _ in range(_n):
        _result.append(random.uniform(0, _max_value))
    return sorted(_result)


def count_membership(tree_data, membership_values, example_index):
    numerator = 0
    denominator = 0
    for key in membership_values:
        key_tuple = tuple(key.split('/'))
        p_yes = tree_data[1][key_tuple].iloc[example_index]
        p_no = tree_data[2][key_tuple].iloc[example_index]

        mu = membership_values.get(key)
        numerator += p_yes * mu
        denominator += (p_yes + p_no) * mu
    if numerator != 0:
        sigma = numerator / denominator
        if sigma > 0.5:
            return 1
        else:
            return 0
    else:
        return 0


def get_accuracy(tree_data):
    data = pd.read_csv('data.csv', sep=',')

    # вместо data должна быть тестовая выборка
    data_array = data.values
    predicted_results = []
    for i in range(len(data_array)):
        # example = data_array[i]
        membership_row = tree_data[0].iloc[i]
        membership_values = {}
        for j in range(len(membership_row)):
            if membership_row[j] > 0:
                memb_tuple = membership_row.index[j]
                memb_index = ''
                memb_tuple_length = len(memb_tuple)
                for k in range(memb_tuple_length):
                    memb_index += memb_tuple[k]
                    if k != memb_tuple_length - 1:
                        memb_index += '/'
                membership_values.update({memb_index: membership_row[j]})
        predicted_results.append(count_membership(tree_data, membership_values, i))
    prediction_model = get_prediction_model(predicted_results, tree_data[3])
    precision = get_precision(prediction_model[0], prediction_model[1])
    recall = get_recall(prediction_model[0], prediction_model[2])

    return f1_score(precision, recall)


def get_prediction_model(predicted_results, true_results):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_normalized = []

    for tr in true_results:
        if tr > 0.5:
            true_normalized.append(1)
        else:
            true_normalized.append(0)
    for i in range(len(predicted_results)):
        if true_normalized[i] == 0 and predicted_results[i] == 1:
            false_positive += 1
        elif true_normalized[i] == 1 and predicted_results[i] == 1:
            true_positive += 1
        elif true_normalized[i] == 1 and predicted_results[i] == 0:
            false_negative += 1

    return [true_positive, false_positive, false_negative]


def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))


def get_precision(tp, fp):
    return tp / (tp + fp)


def get_recall(tp, fn):
    return tp / (tp + fn)


def get_random_couples(array, n):
    winners_array = []
    n_counter = 0

    while n_counter < 20:
        first = random.randint(0, len(array) - 1)
        second = random.randint(0, len(array) - 1)
        if array[first] > array[second]:
            winners_array.append(first)
        else:
            winners_array.append(second)
        n_counter += 1
    return winners_array


def get_new_generation(population, population_priority):
    new_generation = []
    i = 0
    while i < 20:
        index_first = population_priority[i]
        index_second = population_priority[i + 1]
        first_children = mutate(population[index_first], population[index_second])
        second_children = mutate(population[index_second], population[index_first])
        new_generation.append(first_children)
        new_generation.append(second_children)
        i += 2
    return new_generation


def mutate(first_parent, second_parent):
    child = first_parent
    for i in range(len(first_parent)):
        for j in range(len(first_parent[i])):
            term_second = second_parent[i][j][0]
            term_child = child[i][j][0]
            term_child[1] = term_second[1]
            if 0 < j < len(first_parent[i]) - 1:
                term_child[2] = term_second[2]
    return child
