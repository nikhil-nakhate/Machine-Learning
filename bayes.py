import scipy.io.arff as arff
import math
import sys


def parse_data_set():
    train_data_load, metadata_train = arff.loadarff(open(sys.argv[1], 'r'))
    test_data_load, metadata_test = arff.loadarff(open(sys.argv[2], 'r'))

    n_or_t = sys.argv[3]

    return train_data_load, metadata_train, test_data_load, metadata_test, n_or_t


def compute_class_prob():
    class0_count = 0
    class1_count = 0
    for train_inst in train_data_loaded:
        if train_inst[-1] == class0:
            class0_count += 1
        if train_inst[-1] == class1:
            class1_count += 1
    class0_prob_l = float(class0_count + 1) / (len(train_data_loaded) + 2)
    class1_prob_l = float(class1_count + 1) / (len(train_data_loaded) + 2)

    return class0_prob_l, class1_prob_l, class0_count, class1_count

def compute_attr_counts():
    for train_inst in train_data_loaded:
        for attr_idx in range(len(train_inst) - 1):
            attr_values0 = metadata_train_class0_counts[metadata_train.names()[attr_idx]]
            attr_values1 = metadata_train_class1_counts[metadata_train.names()[attr_idx]]
            if train_inst[-1] == class0:
                for attr_value_idx in range(len(attr_values0)):
                    if train_inst[attr_idx] == metadata_train[metadata_train.names()[attr_idx]][-1][attr_value_idx]:
                        attr_values0[attr_value_idx] += 1
            elif train_inst[-1] == class1:
                for attr_value_idx in range(len(attr_values1)):
                    if train_inst[attr_idx] == metadata_train[metadata_train.names()[attr_idx]][-1][attr_value_idx]:
                        attr_values1[attr_value_idx] += 1

    return metadata_train_class0_counts, metadata_train_class1_counts


def separate_classwise(class_value):
    return [train_data_loaded[i] for i in range(len(train_data_loaded)) if train_data_loaded[i][-1] == class_value]


def compute_attr_edges(metadata_train_class0_counts_l, metadata_train_class1_counts_l):
    for attr_idx1_l in range(0, (len(metadata_train.names()) - 1)):
        # while defining range in terms of start stop we have to give the last value
        for attr_idx2_l in range(0, (len(metadata_train.names()) - 1)):
            p_value = 0

            attr_values10 = metadata_train_class0_counts_l[metadata_train.names()[attr_idx1_l]]
            attr_values20 = metadata_train_class0_counts_l[metadata_train.names()[attr_idx2_l]]

            attr_values11 = metadata_train_class1_counts_l[metadata_train.names()[attr_idx1_l]]
            attr_values21 = metadata_train_class1_counts_l[metadata_train.names()[attr_idx2_l]]

            for attr_value1_idx in range(len(metadata_train_list[attr_idx1_l])):
                for attr_value2_idx in range(len(metadata_train_list[attr_idx2_l])):
                    count12 = 0

                    for train_inst in train_data_loaded0:
                        if train_inst[attr_idx1_l] == metadata_train_list[attr_idx1_l][attr_value1_idx] and train_inst[attr_idx2_l] == metadata_train_list[attr_idx2_l][attr_value2_idx]:
                            count12 += 1
                    p_joint = float(count12 + 1) / (len(train_data_loaded) + (len(metadata_train_list[attr_idx1_l]) * len(metadata_train_list[attr_idx2_l]) * 2))
                    p_cond = float(count12 + 1) / (len(train_data_loaded0) + (len(metadata_train_list[attr_idx1_l]) * len(metadata_train_list[attr_idx2_l])))
                    p1 = float(attr_values10[attr_value1_idx]) / (len(train_data_loaded0) + len(attr_values10))
                    p2 = float(attr_values20[attr_value2_idx]) / (len(train_data_loaded0) + len(attr_values20))
                    px1x2 = p_joint * (math.log((p_cond / (p1 * p2)), 2))
                    p_value += px1x2

                    count12 = 0

                    for train_inst in train_data_loaded1:
                        if train_inst[attr_idx1_l] == metadata_train_list[attr_idx1_l][attr_value1_idx] and train_inst[attr_idx2_l] == metadata_train_list[attr_idx2_l][attr_value2_idx]:
                            count12 += 1
                    p_joint = float(count12 + 1) / (len(train_data_loaded) + (len(metadata_train_list[attr_idx1_l]) * len(metadata_train_list[attr_idx2_l]) * 2))
                    p_cond = float(count12 + 1) / (len(train_data_loaded1) + (len(metadata_train_list[attr_idx1_l]) * len(metadata_train_list[attr_idx2_l])))
                    p1 = float(attr_values11[attr_value1_idx]) / (len(train_data_loaded1) + len(attr_values11))
                    p2 = float(attr_values21[attr_value2_idx]) / (len(train_data_loaded1) + len(attr_values21))

                    px1x2 = p_joint * (math.log((p_cond / (p1 * p2)), 2))
                    p_value += px1x2

            if metadata_train.names()[attr_idx1_l] not in edges_dict_weights:
                if metadata_train.names()[attr_idx1_l] == metadata_train.names()[attr_idx2_l]:
                    p_value = -1.0

                edges_dict_weights[metadata_train.names()[attr_idx1_l]] = [[metadata_train.names()[attr_idx2_l], p_value]]
            else:
                if metadata_train.names()[attr_idx1_l] == metadata_train.names()[attr_idx2_l]:
                    p_value = -1.0
                curr_list = edges_dict_weights[metadata_train.names()[attr_idx1_l]]
                curr_list.append([metadata_train.names()[attr_idx2_l], p_value])
                edges_dict_weights[metadata_train.names()[attr_idx1_l]] = curr_list
    return edges_dict_weights


def prims_max_st(edges_dict_weights_l):

    key_array = []
    vertices_array = []
    parent_array = []

    edges_array = []

    for v_idx in range(len(metadata_train.names()) - 1):
        key_array.append(-1.0)
        parent_array.append(None)
        vertices_array.append(metadata_train.names()[v_idx])

    key_array[0] = 1

    for iii in range(len(vertices_array)):
        i1 = key_array.index(max(key_array))
        key_array[i1] = 0
        v1 = vertices_array[i1]
        vertices_array[i1] = None

        for idx_v2 in range(len(edges_dict_weights_l[v1])):
            if edges_dict_weights[v1][idx_v2][0] in vertices_array and edges_dict_weights_l[v1][idx_v2][-1] > key_array[idx_v2]:
                key_array[idx_v2] = edges_dict_weights_l[v1][idx_v2][-1]
                parent_array[idx_v2] = v1

    for idx_p in range(1, len(parent_array)):
        edges_array.append([parent_array[idx_p], metadata_train.names()[idx_p]])

    return edges_array, parent_array


def get_cpts(edges_array_l):
    parents_deno_list = []
    for edge_idx in range(0, len(edges_array_l)):
        parents_value_deno_list = []

        parent_add = metadata_train.names().index(edges_array_l[edge_idx][0])
        for parent_idx_value in range(len(metadata_train_list[parent_add])):
            local_list_deno = []

            count_deno = 0
            for train_inst in train_data_loaded0:
                if train_inst[parent_add] == metadata_train_list[parent_add][parent_idx_value]:
                    count_deno += 1
            local_list_deno.append(count_deno)

            count_deno = 0
            for train_inst in train_data_loaded1:
                if train_inst[parent_add] == metadata_train_list[parent_add][parent_idx_value]:
                    count_deno += 1
            local_list_deno.append(count_deno)
            parents_value_deno_list.append(local_list_deno)
        parents_deno_list.append(parents_value_deno_list)

    final_cpt_list = []
    for edge_idx in range(0, len(edges_array_l)):
        attr_outer_values_list = []

        parent_values_c = len(metadata_train_list[metadata_train.names().index(edges_array_l[edge_idx][0])])
        attr_values_c = len(metadata_train_list[metadata_train.names().index(edges_array_l[edge_idx][1])])

        parent_add = metadata_train.names().index(edges_array_l[edge_idx][0])
        attr_add = metadata_train.names().index(edges_array_l[edge_idx][1])

        for attr_idx_value in range(attr_values_c):
            parent_values_local_list = []
            for parent_idx_value in range(parent_values_c):
                local_list11 = []

                count12 = 0

                for train_inst in train_data_loaded0:
                    if train_inst[parent_add] == metadata_train_list[parent_add][parent_idx_value] and train_inst[attr_add] == metadata_train_list[attr_add][attr_idx_value]:
                        count12 += 1
                p_value = float(count12 + 1) / (parents_deno_list[edge_idx][parent_idx_value][0] + attr_values_c)
                local_list11.append(p_value)

                count12 = 0
                for train_inst in train_data_loaded1:
                    if train_inst[parent_add] == metadata_train_list[parent_add][parent_idx_value] and train_inst[attr_add] == metadata_train_list[attr_add][attr_idx_value]:
                        count12 += 1

                p_value = float(count12 + 1) / (parents_deno_list[edge_idx][parent_idx_value][1] + attr_values_c)
                local_list11.append(p_value)
                parent_values_local_list.append(local_list11)
            attr_outer_values_list.append(parent_values_local_list)
        final_cpt_list.append(attr_outer_values_list)

    attr0_cpt_list = []
    for value0_idx in range(len(metadata_train_list[0])):
        count0 = 0

        local_list00 = []
        for train_inst in train_data_loaded0:
            if train_inst[0] == metadata_train_list[0][value0_idx]:
                count0 += 1
        p_0 = float(count0 + 1) / (len(train_data_loaded0) + len(metadata_train_list[0]))
        local_list00.append(p_0)

        count0 = 0
        for train_inst in train_data_loaded1:
            if train_inst[0] == metadata_train_list[0][value0_idx]:
                count0 += 1
        p_0 = float(count0 + 1) / (len(train_data_loaded1) + len(metadata_train_list[0]))
        local_list00.append(p_0)
        attr0_cpt_list.append(local_list00)

    p_class0 = float(len(train_data_loaded0) + 1) / (len(train_data_loaded) + 2)
    p_class1 = float(len(train_data_loaded1) + 1) / (len(train_data_loaded) + 2)

    return attr0_cpt_list, final_cpt_list, p_class0, p_class1


def predict_tan(attr0_cpt_list_l, final_cpt_list_l, p_class0_l, p_class1_l, parent_array_l):
    for par_idx in range(len(parent_array_l)):
        if par_idx == 0:
            print metadata_train.names()[par_idx], metadata_train.names()[-1]
        else:
            print metadata_train.names()[par_idx], parent_array_l[par_idx], metadata_train.names()[-1]

    print '\n'

    correctly_classified = 0
    for test_inst in test_data_loaded:
        prob0 = 1.0
        prob1 = 1.0
        for attr_idx in range(len(test_inst) - 1):
            if attr_idx == 0:
                for attr_value_idx11 in range(len(metadata_train_list[attr_idx])):
                    if test_inst[attr_idx] == metadata_train_list[attr_idx][attr_value_idx11]:
                        prob0 *= attr0_cpt_list_l[attr_value_idx11][0]
                        prob1 *= attr0_cpt_list_l[attr_value_idx11][1]
            else:
                parent_idx = metadata_train.names().index(parent_array_l[attr_idx])
                for attr_value_idx11 in range(len(metadata_train_list[attr_idx])):
                    for parent_value_idx in range(len(metadata_train_list[parent_idx])):
                        if test_inst[attr_idx] == metadata_train_list[attr_idx][attr_value_idx11] and test_inst[parent_idx] == metadata_train_list[parent_idx][parent_value_idx]:
                            prob0 *= final_cpt_list_l[attr_idx - 1][attr_value_idx11][parent_value_idx][0]
                            prob1 *= final_cpt_list_l[attr_idx - 1][attr_value_idx11][parent_value_idx][1]

        prob0 *= p_class0_l
        prob1 *= p_class1_l

        denom = prob0 + prob1
        prob0 /= denom
        prob1 /= denom

        if prob0 > prob1:
            print class0.replace("'", ""), test_inst[-1].replace("'", ""), "{0:0.12f}".format(prob0)
            if class0 == test_inst[-1]:
                correctly_classified += 1
        elif prob1 > prob0:
            print class1.replace("'", ""), test_inst[-1].replace("'", ""), "{0:0.12f}".format(prob1)
            if class1 == test_inst[-1]:
                correctly_classified += 1

    print '\n', correctly_classified


def predict_nb(prob_class0, prob_class1, count_class0, count_class1, metadata_train_class0_counts_l, metadata_train_class1_counts_l):

    for meta_inst_idx in range(len(metadata_train.names()) - 1):
        print metadata_train.names()[meta_inst_idx], metadata_train.names()[-1]
    print '\n'

    correctly_classified = 0
    for test_inst in test_data_loaded:
        prob0 = 1.0
        prob1 = 1.0
        for attr_idx in range(len(test_inst) - 1):
            attr_values0 = metadata_train_class0_counts_l[metadata_train.names()[attr_idx]]
            attr_values1 = metadata_train_class1_counts_l[metadata_train.names()[attr_idx]]
            for attr_value_idx in range(len(attr_values0)):
                if test_inst[attr_idx] == metadata_test[metadata_test.names()[attr_idx]][-1][attr_value_idx]:
                    prob_class0_attr = float(attr_values0[attr_value_idx]) / (count_class0 + len(attr_values0))
                    prob_class1_attr = float(attr_values1[attr_value_idx]) / (count_class1 + len(attr_values1))
                    prob0 *= prob_class0_attr
                    prob1 *= prob_class1_attr
        prob0 *= prob_class0
        prob1 *= prob_class1
        deno = prob0 + prob1

        prob0 /= deno
        prob1 /= deno

        if prob0 > prob1:
            print class0.replace("'", ""), test_inst[-1].replace("'", ""), "{0:0.12f}".format(prob0)
            if class0 == test_inst[-1]:
                correctly_classified += 1
        elif prob1 > prob0:
            print class1.replace("'", ""), test_inst[-1].replace("'", ""), "{0:0.12f}".format(prob1)
            if class1 == test_inst[-1]:
                correctly_classified += 1

    print '\n', correctly_classified


if __name__ == '__main__':
    global train_data_loaded
    global train_data_loaded0
    global train_data_loaded1

    train_data_loaded, metadata_train, test_data_loaded, metadata_test, n_or_t_main = parse_data_set()

    global edges_dict_weights

    global class0
    global class1

    class0 = metadata_train[metadata_train.names()[-1]][-1][0]
    class1 = metadata_train[metadata_train.names()[-1]][-1][1]

    metadata_train_list = []
    metadata_test_list = []

    metadata_train_class0_counts = {}
    metadata_train_class1_counts = {}

    for metadata_inst in metadata_train:
        if metadata_inst != "class":
            local_list = []
            for attr_value_idx in range(len(metadata_train[metadata_inst][1])):
                local_list.append(metadata_train[metadata_inst][1][attr_value_idx])
            metadata_train_list.append(local_list)
            #initializing with 1 for the Laplace smoothing
            metadata_train_class0_counts[metadata_inst] = [1] * len(metadata_train[metadata_inst][1])
            metadata_train_class1_counts[metadata_inst] = [1] * len(metadata_train[metadata_inst][1])

    train_data_loaded0 = separate_classwise(class0)
    train_data_loaded1 = separate_classwise(class1)

    edges_dict_weights = {}

    class0_prob, class1_prob, class0_count, class1_count = compute_class_prob()
    metadata_train_class0_counts, metadata_train_class1_counts = compute_attr_counts()

    edges_dict_weights_main = compute_attr_edges(metadata_train_class0_counts, metadata_train_class1_counts)
    edges_array_main, parent_array_main = prims_max_st(edges_dict_weights_main)
    attr0_cpt_list_main, final_cpt_list_main, p_class0_main, p_class1_main = get_cpts(edges_array_main)

    if n_or_t_main == "n":
        predict_nb(class0_prob, class1_prob, class0_count, class1_count, metadata_train_class0_counts, metadata_train_class1_counts)

    if n_or_t_main == "t":
        predict_tan(attr0_cpt_list_main, final_cpt_list_main, p_class0_main, p_class1_main, parent_array_main)


