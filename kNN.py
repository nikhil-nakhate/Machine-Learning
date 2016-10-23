import scipy.io.arff as arff
import math
import sys
import operator


def parse_data_set():
    train_data_load, metadata_train = arff.loadarff(open(sys.argv[1], 'r'))
    test_data_load, metadata_test = arff.loadarff(open(sys.argv[2], 'r'))
    k = int(sys.argv[3])

    print train_data_load

    #for u in me

    return train_data_load, metadata_train, test_data_load, metadata_test, k


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
            if metadata_priority_dict[predicted_class] > metadata_priority_dict[t]:
                predicted_class = t
    return predicted_class


def get_prediction_response(neighbours_list):
    response_sum = 0
    for r in range(len(neighbours_list)):
        response_sum += neighbours_list[r][0][-1]
    prediction_response = response_sum / len(neighbours_list)
    return prediction_response


def print_output(train_data, test_data, k, class_or_res):
    print "k Value : ", k
    total_test_instances = len(test_data)
    if class_or_res == "response":
        sum_error = 0
        for i in test_data:
            neighbours_list = get_neighbours(i, train_data, k)
            print "Predicted value : ", "{0:.6f}".format(get_prediction_response(neighbours_list)), '\t', "Actual value : ", "{0:.6f}".format(i[-1])
            sum_error += abs(i[-1] - get_prediction_response(neighbours_list))
        print "Mean absolute error : ", (sum_error / total_test_instances)
        print "Total number of instances : ", total_test_instances
    else:
        sum_rightly_classified = 0
        for i in test_data:
            neighbours_list = get_neighbours(i, train_data, k)
            print "Predicted class : ", get_prediction_class(neighbours_list), '\t', "Actual class : ", i[-1]
            if get_prediction_class(neighbours_list) == i[-1]:
                sum_rightly_classified += 1
        print "Number of correctly classified instances : ", sum_rightly_classified
        print "Total number of instances : ", total_test_instances
        print "Accuracy : ", float (sum_rightly_classified) / total_test_instances


if __name__ == '__main__':
    train_data_loaded, metadata_train, test_data_loaded, metadata_test, k = parse_data_set()
    global metadata_priority_dict
    metadata_priority_dict = {}
    if(metadata_test.names()[-1]) == "class":
        for i in range(len(metadata_train[metadata_test.names()[-1]][1])):
            name_key = metadata_test[metadata_test.names()[-1]][1][i]
            metadata_priority_dict[name_key] = i

    print_output(train_data_loaded, test_data_loaded, k, metadata_test.names()[-1])

