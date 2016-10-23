import sys
import scipy.io.arff as arff
import math


def parse_data_set():
    global train_data
    global train_meta
    train_data, train_meta = arff.loadarff(open(sys.argv[1], 'r'))

    global test_data
    test_data, test_meta = arff.loadarff(open(sys.argv[2], 'r'))

    global m
    m = sys.argv[3]


def get_classes_of_attribute(value_class_pair, attr_value):
    classes = []
    for value in value_class_pair:
        if value[0] == attr_value:
            if value[1] not in classes:
                classes.append(value[1])
        if len(classes) == 2:
            return classes
    return classes


def determine_numeric_candidate_split(train_data_2d, attr_name):
    numeric_candidate_splits = []
    list_of_num_attr = []
    list_of_corresponding_classes = []
    mid_point_array = []
    numeric_candidate_splits.append(attr_name)
    numeric_candidate_splits.append('numeric')
    for row in train_data_2d:
        list_of_num_attr.append(row[train_meta.names().index(attr_name)])
        list_of_corresponding_classes.append(row[-1])

    pair_list = zip(list_of_num_attr, list_of_corresponding_classes)
    pair_list = sorted(pair_list, key=lambda tup: tup[0])
    i = 0
    while (i + 1) < len(pair_list):
        if pair_list[i][0] == pair_list[i + 1][0]:
            i += 1
            continue
        else:
            classes1 = get_classes_of_attribute(pair_list, pair_list[i][0])
            classes2 = get_classes_of_attribute(pair_list, pair_list[i + 1][0])
            for class1 in classes1:
                for class2 in classes2:
                    if class1 != class2:
                        s = (float(pair_list[i][0]) + float(pair_list[i + 1][0])) / 2
                        if s not in mid_point_array:
                            mid_point_array.append((float(pair_list[i][0]) + float(pair_list[i + 1][0])) / 2)
        i += 1

    numeric_candidate_splits.append(mid_point_array)
    return numeric_candidate_splits


def determine_nominal_candidate_split(attr_name):
    nominal_candidate_splits = []
    attr_values = train_meta[attr_name][1]
    nominal_candidate_splits.append(attr_name)
    nominal_candidate_splits.append('nominal')
    nominal_candidate_splits.append(attr_values)
    return nominal_candidate_splits


def determine_candidate_splits(train_data_2d):
    candidate_splits = []
    for j in range(len(train_meta.names())):
        if (train_meta.types()[j]) == 'numeric':
            num_split = determine_numeric_candidate_split(train_data_2d, train_meta.names()[j])
            candidate_splits.append(num_split)
        else:
            nom_split = determine_nominal_candidate_split(train_meta.names()[j])
            candidate_splits.append(nom_split)
    return candidate_splits


def get_info_gain_list(train_data_2d, attr_split_list):
    label_count = class_count(train_data_2d)
    entropy = 0.0
    for count1 in label_count:
        prob = (float(count1) / len(train_data_2d))
        if prob != 0.0:
            entropy -= 1 * prob * math.log(prob, 2)
    info_gain = entropy
    attr_type = attr_split_list[1]
    attr_name = attr_split_list[0]
    if attr_type == 'nominal':
        for i in range(0, len(attr_split_list[2])):
            attr_value = attr_split_list[2][i]
            subset = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'nominal')
            if len(subset) != 0:
                child_class_count_tuple = class_count(subset)
                entropy_child = 0.0
                for count1 in child_class_count_tuple:
                    prob = (float(count1) / len(subset))
                    if prob != 0.0:
                        entropy_child -= 1 * prob * math.log(prob, 2)
            else:
                entropy_child = 0.0
            prob_cond = (float(len(subset)) / len(train_data_2d)) * entropy_child
            info_gain -= prob_cond
        return round(info_gain, 6)
    else:
        info_gain_list = []
        for i in range(0, len(attr_split_list[2])):
            info_gain = entropy
            attr_value = attr_split_list[2][i]
            subset1 = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'left_branch_numeric')
            if len(subset1) != 0:
                child_class_count_left_tuple = class_count(subset1)
                entropy_child1 = 0.0
                for count1 in child_class_count_left_tuple:
                    prob = (float(count1) / len(subset1))
                    if prob != 0.0:
                        entropy_child1 -= 1 * prob * math.log(prob, 2)
            else:
                entropy_child1 = 0.0
            prob_cond = (float(len(subset1)) / len(train_data_2d)) * entropy_child1
            info_gain -= prob_cond

            subset2 = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'right_branch_numeric')
            if len(subset2) != 0:
                child_class_count_right_tuple = class_count(subset2)
                entropy_child2 = 0.0
                for count1 in child_class_count_right_tuple:
                    prob = (float(count1) / len(subset2))
                    if prob != 0.0:
                        entropy_child2 -= 1 * prob * math.log(prob, 2)
            else:
                entropy_child2 = 0.0
            prob_cond = (float(len(subset2)) / len(train_data_2d)) * entropy_child2
            info_gain -= prob_cond
            info_gain_list.append(round(info_gain, 6))
    return info_gain_list


def get_best_split(train_data_2d, candidate_splits):
    max_gain = -1
    info_gain_array = []
    max_gain_tuple = []
    for i in range(0, len(candidate_splits) - 1):
        attr_split_tuple = []
        if candidate_splits[i][1] == 'nominal':
            info_gain = get_info_gain_list(train_data_2d, candidate_splits[i])
            attr_split_tuple.append(info_gain)
            attr_split_tuple.append(candidate_splits[i][0])
            attr_split_tuple.append(candidate_splits[i][1])
            attr_split_tuple.append(candidate_splits[i][2])
        elif len(candidate_splits[i][2]) != 0:
            info_gain = get_info_gain_list(train_data_2d, candidate_splits[i])
            if len(info_gain) == 0:
                attr_split_tuple.append(-1)
                attr_split_tuple.append(candidate_splits[i][0])
                attr_split_tuple.append(candidate_splits[i][1])
                attr_split_tuple.append("")
            else:
                attr_split_tuple.append(max(info_gain))
                attr_split_tuple.append(candidate_splits[i][0])
                attr_split_tuple.append(candidate_splits[i][1])
                max_gain_index = info_gain.index(max(info_gain))
                max_gain_split = candidate_splits[i][2][max_gain_index]
                attr_split_tuple.append(max_gain_split)
        else:
            attr_split_tuple.append(-1)
            attr_split_tuple.append(candidate_splits[i][0])
            attr_split_tuple.append(candidate_splits[i][1])
            attr_split_tuple.append("")
        info_gain_array.append(attr_split_tuple)

    for i in range(0, len(info_gain_array)):
        if info_gain_array[i][0] > max_gain:
            max_gain = info_gain_array[i][0]
            max_gain_tuple = info_gain_array[i]

    if max_gain == -1:
        max_gain_tuple[0] = -1

    return max_gain_tuple


class Node:
    node_type = ""
    node_name = ""
    split_condition_numeric = []
    split_condition_nominal = []
    branches = []

    def __init__(self):
        self.node_type = ""
        self.node_name = ""

    def __str__(self):
        return str(self.node_name)

    def set_name(self, name):
        self.node_name = name

    def get_name(self):
        return self.node_name

    def set_type(self, ntype):
        self.node_type = ntype

    def get_type(self):
        return self.node_type

    def set_split_condition_numeric(self, numeric_sub_tree_type):
        self.split_condition_numeric = numeric_sub_tree_type

    def set_split_condition_nominal(self, split_condition_nominal):
        self.split_condition_nominal = split_condition_nominal

    def get_split_condition_numeric(self):
        return self.split_condition_numeric

    def get_split_condition_nominal(self):
        return self.split_condition_nominal

    def get_branches(self):
        return self.branches


def class_count(train_data_2d):
    count_positive = 0
    count_negative = 0
    for row in train_data_2d:
        if row[train_meta.names().index('class')] == train_meta['class'][1][0]:
            count_negative += 1
        else:
            count_positive += 1
    return count_negative, count_positive


def return_class_value(train_data_2d, parent_class):
    count_negative, count_positive = class_count(train_data_2d)

    if count_negative < count_positive:
        return train_meta['class'][1][1]
    elif count_negative > count_positive:
        return train_meta['class'][1][0]
    else:
        return parent_class


def to_set_parent(count_pair):

    if count_pair[0] < count_pair[1]:
        return train_meta['class'][1][1]
    elif count_pair[0] > count_pair[1]:
        return train_meta['class'][1][0]
    else:
        return 'parent_class'


def check_stopping_criteria_one_class(train_data_2d):
    return_value = 'more_than_one'
    classes_list = []
    for row in train_data_2d:
        classes_list.append(row[-1])
    for i in range(0, len(classes_list) - 1):
        if classes_list[i] == classes_list[i + 1]:
            return_value = classes_list[i]
            continue
        else:
            return 'more_than_one'
    return return_value


def check_stopping_criteria_below_limit(train_data_2d):
    class_count_pair = class_count(train_data_2d)
    if (int(class_count_pair[0]) + int(class_count_pair[1])) < int(m):
        return True
    return False


def subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, numeric_sub_tree_type):
    subset = []
    attr_index = train_meta.names().index(attr_name)

    if attr_type == 'nominal':
        for row in train_data_2d:
            if str(row[attr_index]) == str(attr_value):
                subset.append(row)
    else:
        if 'left_branch_numeric' == numeric_sub_tree_type:
            for row in train_data_2d:
                if float(row[attr_index]) <= float(attr_value):
                    subset.append(row)
        else:
            for row in train_data_2d:
                if float(row[attr_index] > float(attr_value)):
                    subset.append(row)

    return subset


def build_tree(train_data_2d, curr_depth, parent_class):
    branches = []

    if len(train_data_2d) == 0:
        leaf = Node()
        print_class = train_meta['class'][1][0]
        sys.stdout.write(": " + print_class)
        leaf.set_type('leaf')
        return leaf
    one_class = check_stopping_criteria_one_class(train_data_2d)
    if one_class in train_meta['class'][1]:
        leaf = Node()
        leaf.set_name(one_class)
        leaf.set_type('leaf')
        get_return_class = return_class_value(train_data_2d, parent_class)
        sys.stdout.write(": " + str(get_return_class))
        return leaf
    if check_stopping_criteria_below_limit(train_data_2d):
        leaf = Node()
        get_return_class = return_class_value(train_data_2d, parent_class)
        sys.stdout.write(": " + str(get_return_class))
        leaf.set_name(get_return_class)
        leaf.set_type('leaf')
        return leaf
    else:
        candidate_splits = determine_candidate_splits(train_data_2d)
        max_gain_tuple = get_best_split(train_data_2d, candidate_splits)
        if max_gain_tuple[0] == -1:
            m = str(return_class_value(train_data_2d, parent_class))
            sys.stdout.write(": " + m)
            leaf = Node()
            leaf.set_name(m)
            leaf.set_type('leaf')
            return leaf
        root = Node()
        root.set_name(max_gain_tuple[1])
        root.set_type(max_gain_tuple[2])
        if max_gain_tuple[2] == 'nominal':
            root.set_split_condition_nominal(max_gain_tuple[3])
            for i in range(0, len(max_gain_tuple[3])):
                attr_name = max_gain_tuple[1]
                attr_value = max_gain_tuple[3][i]
                attr_type = max_gain_tuple[2]
                subset = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'nominal')
                child_class_count_tuple = class_count(subset)
                get_return_class = to_set_parent(child_class_count_tuple)
                if get_return_class == 'parent_class':
                    get_return_class = parent_class

                num_of_tabs = 0
                print "\r\n",
                while num_of_tabs < curr_depth:
                    print '|\t',
                    num_of_tabs += 1
                sys.stdout.write(attr_name + " = " + attr_value)
                sys.stdout.write(" [")
                print str(child_class_count_tuple[0]) + " ",
                sys.stdout.write(str(child_class_count_tuple[1]))
                sys.stdout.write("]")
                child = build_tree(subset, curr_depth + 1, get_return_class)
                branches.append(child)
            curr_depth += 1

        else:
            attr_name = max_gain_tuple[1]
            attr_value = max_gain_tuple[3]
            attr_type = max_gain_tuple[2]
            root.set_split_condition_numeric(attr_value)
            subset1 = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'left_branch_numeric')
            child_class_count_tuple = class_count(subset1)
            get_return_class1 = to_set_parent(child_class_count_tuple)
            if get_return_class1 == 'parent_class':
                get_return_class1 = parent_class
            num_of_tabs = 0
            print "\r\n",
            while num_of_tabs < curr_depth:
                print "|\t",
                num_of_tabs += 1
            print max_gain_tuple[1],
            sys.stdout.write(" <= %0.6f " % attr_value)
            sys.stdout.write("[")
            print str(child_class_count_tuple[0]) + " ",
            sys.stdout.write(str(child_class_count_tuple[1]))
            sys.stdout.write("]")
            subset2 = subset_for_branch(train_data_2d, attr_name, attr_value, attr_type, 'right_branch_numeric')
            child_class_count_tuple = class_count(subset2)
            get_return_class2 = to_set_parent(child_class_count_tuple)
            if get_return_class2 == 'parent_class':
                get_return_class2 = parent_class
            temp_curr_depth = curr_depth + 1
            child1 = build_tree(subset1, temp_curr_depth, get_return_class1)
            branches.append(child1)
            num_of_tabs = 0
            print "\r\n",
            while (num_of_tabs < curr_depth):
                print "|\t",
                num_of_tabs += 1
            print max_gain_tuple[1],
            sys.stdout.write(" > %0.6f " % attr_value)
            sys.stdout.write("[")
            print str(child_class_count_tuple[0]) + " ",
            sys.stdout.write(str(child_class_count_tuple[1]))
            sys.stdout.write("]")
            child2 = build_tree(subset2, temp_curr_depth, get_return_class2)
            branches.append(child2)
        root.branches = branches
        return root


def predictions_on_test_data(instance, tree, instance_num):
    attr_name = tree.get_name()
    global correct
    global wrong
    if tree.get_type() == 'leaf':
        if tree.get_name() == instance[-1]:
            correct += 1
        else:
            wrong += 1
        print("%3d: Actual: " % (instance_num) + str(instance[-1]) + " Predicted: " + str(tree.get_name()))
    elif tree.get_type() == 'nominal':
        j = 0
        for split in tree.get_split_condition_nominal():
            if instance[train_meta.names().index(attr_name)] == split:
                predictions_on_test_data(instance, tree.branches[j], instance_num)
                break
            j += 1
    else:
        if instance[train_meta.names().index(attr_name)] <= tree.get_split_condition_numeric():
            predictions_on_test_data(instance, tree.branches[0], instance_num)
        else:
            predictions_on_test_data(instance, tree.branches[1], instance_num)


if __name__ == '__main__':
    correct = 0
    wrong = 0
    
    parse_data_set()

    tree = Node()
    parent_class = train_meta['class'][1][0]
    tree = build_tree(train_data, 0, parent_class)
    print '\n<Predictions for the Test Set Instances>'
    instance_num = 1
    for row in test_data:
        predictions_on_test_data(row, tree, instance_num)
        instance_num += 1
    print "Number of correctly classified: " + str(correct) + " Total number of test instances: " + str(wrong + correct)