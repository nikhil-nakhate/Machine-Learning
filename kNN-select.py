import scipy.io.arff as arff
import numpy as np
import math
import sys
import operator


def parse_data_set():
    train_data_load, metadata_train = arff.loadarff(open(sys.argv[1], 'r'))
    test_data_load, metadata_test = arff.loadarff(open(sys.argv[2], 'r'))
    k1 = int(sys.argv[3])
    k2 = int(sys.argv[4])
    k3 = int(sys.argv[5])

    return train_data_load, metadata_train, test_data_load, metadata_test, k1, k2, k3


def euclidean_distance(test_instance, one_train_instance, features_only_length):
    dist_square = 0
    for f in range(features_only_length):
        dist_square += pow((test_instance[f] - one_train_instance[f]), 2)
    dist = math.sqrt(dist_square)
    return dist


def get_neighbours(test_instance, train_set, k):
    distances_list = []
    neighbours_list = []
    num_of_features = len(test_instance) - 1
    for i in range(len(train_set)):
        dist_value = euclidean_distance(test_instance, train_set[i], num_of_features)
        distances_list.append((train_set[i], dist_value))
    distances_list.sort(key=operator.itemgetter(1))
    for j in range(k):
        neighbours_list.append(distances_list[j])
    return neighbours_list


def get_neighbours_knn(test_index, train_set, k):
    distances_list = []
    neighbours_list = []
    num_of_features = len(train_set[test_index]) - 1
    for i in range(len(train_set)):
        if i != test_index:
            dist_value = euclidean_distance(train_set[test_index], train_set[i], num_of_features)
            distances_list.append((train_set[i], dist_value))
    distances_list.sort(key=operator.itemgetter(1))
    for j in range(k):
        neighbours_list.append(distances_list[j])
    return neighbours_list


def get_prediction_class(neighbours_list):
    response_dict = {}
    for r in range(len(neighbours_list)):
        class_value = neighbours_list[r][0][-1]
        if class_value in response_dict:
            response_dict[class_value] += 1
        else:
            response_dict[class_value] = 1

    prediction_class_tuple = max(response_dict.iteritems(), key=operator.itemgetter(1))
    predicted_class = prediction_class_tuple[0]
    for t in response_dict.keys():
        if prediction_class_tuple[1] == response_dict[t]:
            if metadata_priority_dict_train[predicted_class] > metadata_priority_dict_train[t]:
                predicted_class = t
    return predicted_class


def get_prediction_response(neighbours_list):
    response_sum = 0
    for r in range(len(neighbours_list)):
        response_sum += neighbours_list[r][0][-1]
    prediction_response = response_sum / len(neighbours_list)
    return prediction_response


def find_accuracy_error(train_data, k, class_or_res):
    total_test_instances = len(train_data)
    if class_or_res == "response":
        sum_error = 0

        for i in range(len(train_data)):
            neighbours_list = get_neighbours_knn(i, train_data, k)
            sum_error += abs(train_data[i][-1] - get_prediction_response(neighbours_list))
        mean_absolute_error = (sum_error / total_test_instances)
        return mean_absolute_error
    else:
        sum_rightly_classified = 0
        for i in range(len(train_data)):
            neighbours_list = get_neighbours_knn(i, train_data, k)
            if get_prediction_class(neighbours_list) == train_data[i][-1]:
                sum_rightly_classified += 1
        accuracy = float (sum_rightly_classified) / total_test_instances
        sum_wrongly_classified = total_test_instances - sum_rightly_classified
        return sum_wrongly_classified


def print_output(train_data, test_data, k, class_or_res):
    total_test_instances = len(test_data)
    if class_or_res == "response":
        sum_error = 0
        for i in range(len(test_data)):
            neighbours_list = get_neighbours(test_data[i], train_data, k)
            print "Predicted value : ", "{0:.6f}".format(get_prediction_response(neighbours_list)), '\t', "Actual value : ", "{0:.6f}".format(test_data[i][-1])
            sum_error += abs(test_data[i][-1] - get_prediction_response(neighbours_list))
        print "Mean absolute error : ", (sum_error / total_test_instances)
        print "Total number of instances : ", total_test_instances
    else:
        sum_rightly_classified = 0
        for i in range(len(test_data)):
            neighbours_list = get_neighbours(test_data[i], train_data, k)
            print "Predicted class : ", get_prediction_class(neighbours_list), '\t', "Actual class : ", test_data[i][-1]
            if get_prediction_class(neighbours_list) == test_data[i][-1]:
                sum_rightly_classified += 1
        print "Number of correctly classified instances : ", sum_rightly_classified
        print "Total number of instances : ", total_test_instances
        print "Accuracy : ", float (sum_rightly_classified) / total_test_instances


if __name__ == '__main__':
    train_data_loaded, metadata_train, test_data_loaded, metadata_test, k1, k2, k3 = parse_data_set()
    global metadata_priority_dict_train
    metadata_priority_dict_train = {}
    if(metadata_train.names()[-1]) == "class":
        for i in range(len(metadata_train[metadata_train.names()[-1]][1])):
            name_key = metadata_train[metadata_train.names()[-1]][1][i]
            metadata_priority_dict_train[name_key] = i

    global metadata_priority_dict_test
    metadata_priority_dict_test = {}
    if(metadata_test.names()[-1]) == "class":
        for i in range(len(metadata_train[metadata_test.names()[-1]][1])):
            name_key = metadata_test[metadata_test.names()[-1]][1][i]
            metadata_priority_dict_test[name_key] = i

    if metadata_test.names()[-1] == "response":
        mean_abs_error1 = find_accuracy_error(train_data_loaded, k1, metadata_train.names()[-1])
        print "Mean absolute error for k = ", k1, " : ", mean_abs_error1
        mean_abs_error2 = find_accuracy_error(train_data_loaded, k2, metadata_train.names()[-1])
        print "Mean absolute error for k = ", k2, " : ", mean_abs_error2
        mean_abs_error3 = find_accuracy_error(train_data_loaded, k3, metadata_train.names()[-1])
        print "Mean absolute error for k = ", k3, " : ", mean_abs_error3

        value_dict1 = {mean_abs_error1: k1, mean_abs_error2: k2, mean_abs_error3: k3}

        min_value = min(mean_abs_error1, mean_abs_error2, mean_abs_error3)
        print "Best k value : ", value_dict1[min_value]
        print_output(train_data_loaded, test_data_loaded, value_dict1[min_value], "response")
    else:
        incorrectly_classified1 = find_accuracy_error(train_data_loaded, k1, metadata_train.names()[-1])
        print "Number of incorrectly classified instances for k = ", k1, " : ", incorrectly_classified1
        incorrectly_classified2 = find_accuracy_error(train_data_loaded, k2, metadata_train.names()[-1])
        print "Number of incorrectly classified instances for k = ", k2, " : ", incorrectly_classified2
        incorrectly_classified3 = find_accuracy_error(train_data_loaded, k3, metadata_train.names()[-1])
        print "Number of incorrectly classified instances for k = ", k3, " : ", incorrectly_classified3
        value_dict = {incorrectly_classified1 : k1, incorrectly_classified2 : k2, incorrectly_classified3 : k3}

        min_value = min(incorrectly_classified1, incorrectly_classified2, incorrectly_classified3)
        print "Best k value : ", value_dict[min_value]
        print_output(train_data_loaded, test_data_loaded, value_dict[min_value], "class")
    #print_output(train_data_loaded, test_data_loaded, k, metadata_test.names()[-1])

