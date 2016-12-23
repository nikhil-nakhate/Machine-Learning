import scipy.io.arff as arff
import math
import sys
import numpy as np


def parse_data_set():
    l = float(sys.argv[1])
    h = int(sys.argv[2])
    e = int(sys.argv[3])

    train_data_load, metadata_train = arff.loadarff(open(sys.argv[4], 'r'))
    test_data_load, metadata_test = arff.loadarff(open(sys.argv[5], 'r'))

    return l, h, e, train_data_load, metadata_train, test_data_load, metadata_test


def normlize_numeric_features(data_set_load, metadata_local):
    mean_array_local = []
    stddev_array_local = []
    for index_attr in range(len(metadata_local.names())):
        sum2 = 0
        mean_array_local.append(0)
        stddev_array_local.append(0)
        if metadata_local[metadata_local.names()[index_attr]][1] is None:
            for train_inst in data_set_load:
                sum2 = sum2 + train_inst[index_attr]

            mean = sum2 / (len(data_set_load))

            mean_array_local[index_attr] = mean
            sq_sum = 0
            for train_inst in data_set_load:
                sq_sum += pow((train_inst[index_attr] - mean), 2)

            div_by_size = sq_sum / (len(data_set_load))
            stddev = math.sqrt(div_by_size)
            stddev_array_local[index_attr] = stddev
    return mean_array_local, stddev_array_local


def normlize_numeric_features_test(test_inst, metadata_local):
    value2_array = []
    for index_attr in range(len(test_inst)):
        value2_array.append(test_inst[index_attr])
        if metadata_local[metadata_local.names()[index_attr]][1] is None:
            value2 = (test_inst[index_attr] - mean_array[index_attr]) / stddev_array[index_attr]
            value2_array[index_attr] = (value2)
    return value2_array


def normlize_numeric_features_train(train_inst, metadata_local):
    value1_array = []
    for index_attr in range(len(train_inst)):
        value1_array.append(train_inst[index_attr])
        if metadata_local[metadata_local.names()[index_attr]][1] is None:
            value1 = (train_inst[index_attr] - mean_array[index_attr]) / stddev_array[index_attr]
            value1_array[index_attr] = value1
    return value1_array


def encoded_input_array(data_set_instance, metadata_local, metadata_dict_local):
    attr_encoded_array = []
    for index_of_attr in range(len(data_set_instance)):
        if (metadata_dict_local[metadata_local.names()[index_of_attr]] is None) and (index_of_attr != len(data_set_instance) - 1):
            attr_encoded_array.append(data_set_instance[index_of_attr])
        elif index_of_attr != (len(data_set_instance) - 1):
            num_of_values_of_attr = len(metadata_dict_local[metadata_train.names()[index_of_attr]])
            input_binary_encode = [0] * num_of_values_of_attr
            index_curr_value = metadata_dict_local[metadata_train.names()[index_of_attr]].index(data_set_instance[index_of_attr])
            input_binary_encode[index_curr_value] = 1
            for index1 in range(len(input_binary_encode)):
                attr_encoded_array.append(input_binary_encode[index1])

    return attr_encoded_array


def sigmoid_function(weight_array_sig, attr_encoded_sig, bias):
    sum1 = 0
    for index_w_a in range(len(weight_array_sig)):
        sum23 = sum1 + (weight_array_sig[index_w_a] * attr_encoded_sig[index_w_a])
        sum1 = sum23

    net_value = sum1 + bias
    take_neg_net_value = (-1) * net_value

    exp_value = math.exp(take_neg_net_value)

    sig_value = 1 / (1 + exp_value)

    return sig_value


def update_weights(l, weight_array_curr, delta_inst_curr, data_set_inst_curr, bias_curr):
    weight_array_new = []
    for index_w in range(len(weight_array_curr)):
        delta_weight = l * delta_inst_curr * data_set_inst_curr[index_w]
        weight_new = weight_array_curr[index_w] + delta_weight
        weight_array_new.append(weight_new)

    delta_bias_curr = l * delta_inst_curr
    bias_curr = bias_curr + delta_bias_curr

    return bias_curr, weight_array_new


def output_encode(data_set_instance, metadata_local):
    if data_set_instance[-1] == metadata_local[metadata_train.names()[-1]][1][0]:
        return 0
    else:
        return 1


def get_random_weights(size):
    bias = np.random.uniform(-0.01, 0.010000000000001)
    random_weight_arr = []
    for index_w_arr in range(size):
        random_weight_arr.append(np.random.uniform(-0.01, 0.010000000000001))

    return bias, random_weight_arr


def cross_entropy_error_inst(final_value, actual_value):
    error_inst = (-1 * actual_value * math.log(final_value)) - ((1 - actual_value) * math.log((1 - final_value)))
    return error_inst


def print_test_one_epoch(bias_ipop11, weight_array_ipop11, bias_hop11, weight_array_hop11, full_hid_iph11, full_bias_hid_iph11):
    cross_entropy_error_final_tt = 0
    correctly_classified_tt = 0
    incorrectly_classified_tt = 0
    for index_data_set2 in range(len(test_data_loaded)):
        op_value_main = output_encode(test_data_loaded[index_data_set2], metadata_test)
        enoded_ip_arr = encoded_input_array(normlize_numeric_features_test(test_data_loaded[index_data_set2], metadata_test), metadata_test,
            metadata_test_dict)
        if h == 0:

            sig_value_main = sigmoid_function(weight_array_ipop11, enoded_ip_arr, bias_ipop11)

            if sig_value_main > 0.5:
                op_predicted = 1
            else:
                op_predicted = 0

            if op_value_main == op_predicted:
                correctly_classified_tt += 1
            else:
                incorrectly_classified_tt += 1

            cross_entropy_error_final_tt += cross_entropy_error_inst(sig_value_main, op_value_main)

            print "{0:0.12f}".format(sig_value_main), '\t\t', metadata_test[metadata_test.names()[-1]][1][op_predicted], '\t', metadata_test[metadata_train.names()[-1]][1][op_value_main]
        else:
            hid_output_values = []
            for index_hid_units in range(h):
                sig_value_hid = sigmoid_function(full_hid_iph11[index_hid_units], enoded_ip_arr, full_bias_hid_iph11[index_hid_units])
                hid_output_values.append(sig_value_hid)

            sig_value_op_main = sigmoid_function(weight_array_hop11, hid_output_values, bias_hop11)

            delta_output = op_value_main - sig_value_op_main

            delta_hid_array = []
            for index_hid_units_err in range(h):
                delta_inst_curr = hid_output_values[index_hid_units_err] * (1 - hid_output_values[index_hid_units_err]) * delta_output * weight_array_hop11[index_hid_units_err]
                delta_hid_array.append(delta_inst_curr)

            for index_hid_units_upd in range(h):
                full_bias_hid_iph11[index_hid_units_upd], full_hid_iph11[index_hid_units_upd] = update_weights(l, full_hid_iph11[index_hid_units_upd], delta_hid_array[index_hid_units_upd],
                                                                                                               enoded_ip_arr, full_bias_hid_iph11[index_hid_units_upd])

            if sig_value_op_main > 0.5:
                op_predicted = 1
            else:
                op_predicted = 0

            if op_value_main == op_predicted:
                correctly_classified_tt += 1
            else:
                incorrectly_classified_tt += 1

            cross_entropy_error_final_tt += cross_entropy_error_inst(sig_value_op_main, op_value_main)

            print "{0:0.12f}".format(sig_value_op_main), '\t\t', metadata_test[metadata_train.names()[-1]][1][op_predicted], '\t', metadata_test[metadata_train.names()[-1]][1][op_value_main]

    return cross_entropy_error_final_tt, correctly_classified_tt, incorrectly_classified_tt


def print_train_one_epoch(bias_ipop13, weight_array_ipop13, bias_hop13, weight_array_hop13, full_hid_iph13, full_bias_hid_iph13):
    cross_entropy_error_train = 0
    correctly_classified_t = 0
    incorrectly_classified_t = 0
    np.random.shuffle(train_data_loaded)
    for index_data_set1 in range(len(train_data_loaded)):
        op_value_main = output_encode(train_data_loaded[index_data_set1], metadata_train)
        enoded_ip_arr = encoded_input_array(
             normlize_numeric_features_train(train_data_loaded[index_data_set1], metadata_train), metadata_train,
             metadata_train_dict)
        if h == 0:
            sig_value_main = sigmoid_function(weight_array_ipop13, enoded_ip_arr, bias_ipop13)

            delta_inst_curr = op_value_main - sig_value_main

            if sig_value_main > 0.5:
                op_predicted = 1
            else:
                op_predicted = 0

            if op_value_main == op_predicted:
                correctly_classified_t += 1
            else:
                incorrectly_classified_t += 1

            bias_ipop2, weight_array_ipop2 = update_weights(l, weight_array_ipop13, delta_inst_curr, enoded_ip_arr,
                                                            bias_ipop13)

            bias_ipop13 = bias_ipop2
            for index_c in range(len(weight_array_ipop2)):
                weight_array_ipop13[index_c] = weight_array_ipop2[index_c]

            cross_entropy_error_curr = cross_entropy_error_inst(sig_value_main, op_value_main)
            cross_entropy_error_train += cross_entropy_error_curr
        else:
            hid_output_values = []
            delta_hid_array = []
            for index_hid_units in range(h):
                sig_value_hid = sigmoid_function(full_hid_iph13[index_hid_units], enoded_ip_arr, full_bias_hid_iph13[index_hid_units])
                hid_output_values.append(sig_value_hid)

            sig_value_op_main = sigmoid_function(weight_array_hop13, hid_output_values, bias_hop13)

            delta_output = op_value_main - sig_value_op_main

            for index_hid_units_err in range(h):
                delta_inst_curr = hid_output_values[index_hid_units_err] * (1 - hid_output_values[index_hid_units_err]) * delta_output * weight_array_hop13[index_hid_units_err]
                delta_hid_array.append(delta_inst_curr)

            bias_hop1, weight_array_hop1 = update_weights(l, weight_array_hop13, delta_output, hid_output_values,
                                                          bias_hop13)

            for index_hid_units_upd in range(h):
                full_bias_hid_iph13[index_hid_units_upd], full_hid_iph13[index_hid_units_upd] = update_weights(l, full_hid_iph13[index_hid_units_upd], delta_hid_array[index_hid_units_upd],
                                                                                                               enoded_ip_arr, full_bias_hid_iph13[index_hid_units_upd])

            bias_hop13 = bias_hop1
            for index_w1 in range(len(weight_array_hop1)):
                weight_array_hop13[index_w1] = weight_array_hop1[index_w1]

            if sig_value_op_main > 0.5:
                op_predicted = 1
            else:
                op_predicted = 0

            if op_value_main == op_predicted:
                correctly_classified_t += 1
            else:
                incorrectly_classified_t += 1

            cross_entropy_error_curr = cross_entropy_error_inst(sig_value_op_main, op_value_main)
            cross_entropy_error_train += cross_entropy_error_curr

    return cross_entropy_error_train, correctly_classified_t, incorrectly_classified_t


if __name__ == '__main__':
    l, h, e, train_data_loaded, metadata_train, test_data_loaded, metadata_test = parse_data_set()
    global metadata_train_dict
    global metadata_test_dict

    global mean_array
    global stddev_array

    metadata_train_dict = {}
    metadata_test_dict = {}

    mean_array, stddev_array = normlize_numeric_features(train_data_loaded, metadata_train)

    for metadata_inst in metadata_train:
        if metadata_train[metadata_inst][0] == "nominal":
            metadata_train_dict[metadata_inst] = metadata_train[metadata_inst][1]
        else:
            metadata_train_dict[metadata_inst] = None

    for metadata_inst in metadata_test:
        if metadata_test[metadata_inst][0] == "nominal":
            metadata_test_dict[metadata_inst] = metadata_test[metadata_inst][1]
        else:
            metadata_test_dict[metadata_inst] = None

    bias_ipop, weight_array_ipop = get_random_weights(len(encoded_input_array(train_data_loaded[0], metadata_train, metadata_train_dict)))

    bias_hop, weight_array_hop = get_random_weights(h)

    full_hid_iph = []
    full_bias_hid_iph = []

    for index_this in range(h):
        bias_iph, weight_array_iph = get_random_weights(len(encoded_input_array(train_data_loaded[0], metadata_train, metadata_train_dict)))
        full_hid_iph.append(weight_array_iph)
        full_bias_hid_iph.append(bias_iph)

    for index_epoch in range(e):
        cross_entropy_error_final, correctly_classified, incorrectly_classified = print_train_one_epoch(bias_ipop, weight_array_ipop, bias_hop, weight_array_hop, full_hid_iph, full_bias_hid_iph)

        print index_epoch + 1, '\t\t', "{0:0.12f}".format(cross_entropy_error_final), '\t', correctly_classified, '\t', incorrectly_classified

    cross_entropy_error_final_test, correctly_classified_test, incorrectly_classified_test = print_test_one_epoch(bias_ipop, weight_array_ipop, bias_hop, weight_array_hop, full_hid_iph, full_bias_hid_iph)
    print "Correctly Classified:", correctly_classified_test, "Incorrectly Classified:", incorrectly_classified_test
