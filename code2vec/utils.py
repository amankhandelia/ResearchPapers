import tensorflow as tf
import numpy as np
import json
import os
import pickle as pkl
# import matplotlib.pyplot as plt

def get_sequence_example(sequence, label):
    """
    Serializes a pair of sequence, label for the storage in tfrecord

    :param sequence: a set of path_context (where order does not matter) as defined in code2vec paper
    :param label: method name which is being represented by the given path_context
    :return: stores them in SequenceExample and then returns the same, hence represent a single datapoint
    """
    datapoint = tf.train.SequenceExample()
    # A non-sequential feature of our example
    sequence_length = len(sequence)
    datapoint.context.feature["length"].int64_list.value.append(sequence_length)
    # Feature lists for the two sequential features of our example
    # fl_tokens = datapoint.feature_lists.feature_list["tokens"]
    feature_sequence = datapoint.feature_lists.feature_list["sequence"]
    feature_labels = datapoint.feature_lists.feature_list["labels"]
    for token in sequence:
        feature_sequence.feature.add().int64_list.value.extend(token)
    feature_labels.feature.add().int64_list.value.append(label)
    return datapoint

def parse_sequence(datapoint_raw):
    """
    Deserializes a datapoint (sequence, label) and returns the same
    :param datapoint_raw: Serialized datapoint
    :return: Deserialized datapoint
    """
    context_features = {
        "length":tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_feature = {
        "sequence": tf.FixedLenSequenceFeature([3], dtype=tf.int64),
        "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }
    datapoint = tf.parse_single_sequence_example(datapoint_raw, context_features=context_features, sequence_features=sequence_feature)
    return datapoint

def write_tfRecord(sequences, labels, file_path):
    """
    Writes down the given list of labels and sequences to tfrecord at the given path

    :param sequences: A list of sequence where each sequence is defined as a set of path-context
    :param labels: A list of label which are essentially the name of the function
    :param file_path: Path where the file is to be persisted
    :return: None
    """
    writer = tf.python_io.TFRecordWriter(path=file_path)
    for sequence, label in zip(sequences, labels):
        datapoint = get_sequence_example(sequence, label).SerializeToString()
        writer.write(datapoint)
    # writer.flush()
    writer.close()

def inflate(x, y):
    """
    A intermediary function whose job is to handle the subtle complexties related to padding a batch
    :param x:
    :param y:
    :return:
    """
    x['length'] = tf.expand_dims(tf.convert_to_tensor(x['length']),0)
    return x, y
def deflate(x, y):
    """
    A intermediary function whose job is to handle the subtle complexties related to padding a batch.
    :param x:
    :param y:
    :return:
    """
    x['length'] = tf.squeeze(x['length'])
    y['labels'] = tf.squeeze(y['labels'])
    return x, y

def create_dictionaries(ast_representation_directory, value_vocab_cutoff = -1, max_sequence_length = 5000):
    """
    Creates a dictionary of all the paths and values, as defined in code2vec paper, using the json file which follows below mentioned format.
    This method recursively traverses all the directories and reads the non-empty json files and returns dictionaries created

    JSON Format:

    [{methodRepresentation}, {methodRepresentation}, {methodRepresentation},......., {methodRepresentation}]

    methodRepresentation:{
    name:methodName
    paths:[{pathRepresentation},{pathRepresentation},{pathRepresentation},.....,{pathRepresentation}]
    }

    pathRepresentation:{
    path:PathString
    ValueLeft: SomeValue
    ValueRight: SomeValue
    }

    :param ast_representation_directory:
    :return:
    """
    path_vocab, value_vocab, tag_vocab ={}, {}, {}
    count, counter = 0, 0
    for dirpath, dnames, fnames in os.walk(ast_representation_directory):
        for f in fnames:
            if f.endswith(".json"):
                if counter%1000==0: print(counter)
                if os.stat(os.path.join(dirpath, f)).st_size > 0:

                    json_data = open(os.path.join(dirpath, f), 'r')
                    json_data = json.load(json_data)
                    for method_representation in json_data:
                        counter += 1
                        if len(method_representation['paths']) <= max_sequence_length:
                            if method_representation['name'] not in tag_vocab:
                                tag_vocab[method_representation['name']] = method_representation['name']
                            for path in method_representation['paths']:
                                if path['path'] not in path_vocab:
                                    path_vocab[path['path']] = path['path']
                                if path['valueLeft'] not in value_vocab:
                                    value_vocab[path['valueLeft']] = path['valueLeft']
                                if path['valueRight'] not in value_vocab:
                                    value_vocab[path['valueRight']] = path['valueRight']
                        if value_vocab_cutoff != -1 and len(value_vocab) > value_vocab_cutoff:
                            return (path_vocab, value_vocab, tag_vocab, counter)
                else:
                    count+=1
    print(count, counter)
    return (path_vocab, value_vocab, tag_vocab)

def generate_and_save_dict(path_to_dataset_json, path_to_dictionary_location):
    """

    :param path_to_dataset_json:
    :param path_to_dictionary_location:
    :return:
    """
    path_vocab, value_vocab, tag_vocab = create_dictionaries(path_to_dataset_json)
    for i, j in zip(path_vocab, range(1, len(path_vocab) + 1)):
        path_vocab[i] = j
    for i, j in zip(value_vocab, range(1, len(value_vocab) + 1)):
        value_vocab[i] = j
    for i, j in zip(tag_vocab, range(1, len(tag_vocab)+1)):
        tag_vocab[i] = j
    pkl.dump(path_vocab, open(os.path.join(path_to_dictionary_location, 'path_vocab.pkl'), 'wb'))
    pkl.dump(value_vocab, open(os.path.join(path_to_dictionary_location, 'value_vocab.pkl'), 'wb'))
    pkl.dump(tag_vocab, open(os.path.join(path_to_dictionary_location, 'tag_vocab.pkl'), 'wb'))
    value_path_vocab = {}
    for i, j in zip(value_vocab, range(1, len(value_vocab) + 1)):
        value_path_vocab[i] = j
    for i, j in zip(path_vocab, range(len(value_vocab) + 1, len(value_vocab) + len(path_vocab) + 1)):
        value_path_vocab[i] = j
    with open(os.path.join(path_to_dictionary_location, 'path_and_value_vocab.pkl'), 'wb') as file:
        pkl.dump(value_path_vocab, file)

def modify_dict(path_to_dictionary_location):
    """

    :param path_to_dictionary_location:
    :return:
    """
    path_vocab = pkl.load(open(os.path.join(path_to_dictionary_location, 'path_vocab.pkl'), 'rb'))
    value_vocab = pkl.load(open(os.path.join(path_to_dictionary_location, 'value_vocab.pkl'), 'rb'))
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionary_location, 'tag_vocab.pkl'), 'rb'))
    for i, j in zip(path_vocab, range(0, len(path_vocab))):
        path_vocab[i] = j
    for i, j in zip(value_vocab, range(0, len(value_vocab))):
        value_vocab[i] = j
    for i, j in zip(tag_vocab, range(0, len(tag_vocab))):
        tag_vocab[i] = j
    pkl.dump(path_vocab, open(os.path.join(path_to_dictionary_location, 'path_vocab.pkl'), 'wb'))
    pkl.dump(value_vocab, open(os.path.join(path_to_dictionary_location, 'value_vocab.pkl'), 'wb'))
    pkl.dump(tag_vocab, open(os.path.join(path_to_dictionary_location, 'tag_vocab.pkl'), 'wb'))

def get_datapoints(path_to_dictionaries, path_to_dataset, max_seq_length):
    """

    :param path_to_dictionaries:
    :param path_to_dataset:
    :param max_seq_length:
    :return:
    """
    path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_vocab.pkl'), 'rb'))
    value_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'value_vocab.pkl'), 'rb'))
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    sequences = []
    labels = []
    for dirpath, dnames, fnames in os.walk(path_to_dataset):
        for f in fnames:
            if f.endswith(".json"):
                #The condition below ensures that it does not tries read a empty file
                if os.stat(os.path.join(dirpath, f)).st_size > 0:
                    json_data = open(os.path.join(dirpath, f), 'r')
                    json_data = json.load(json_data)
                    for method_representation in json_data:
                        labels.append(tag_vocab[method_representation['name']])
                        sequence = []
                        for path in method_representation['paths']:
                            sequence.append([value_vocab[path['valueLeft']], path_vocab[path['path']], value_vocab[path['valueRight']]])
                        if len(sequence) <= max_seq_length:
                            sequences.append(sequence)
    return sequences, labels

def get_datapoints_combined(path_to_dictionaries, path_to_dataset, max_seq_length):
    """

    :param path_to_dictionaries:
    :param path_to_dataset:
    :param max_seq_length:
    :return:
    """
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    value_and_path_vocab = {}
    with open(os.path.join(path_to_dictionaries, 'path_and_value_vocab.pkl'), 'rb') as file:
        value_and_path_vocab = pkl.load(file)
    # print(value_and_path_vocab)
    sequences = []
    labels = []
    for dirpath, dnames, fnames in os.walk(path_to_dataset):
        for f in fnames:
            if f.endswith(".json"):
                #The condition below ensures that it does not tries read a empty file
                if os.stat(os.path.join(dirpath, f)).st_size > 0:
                    json_data = open(os.path.join(dirpath, f), 'r')
                    json_data = json.load(json_data)
                    for method_representation in json_data:
                        labels.append(tag_vocab[method_representation['name']])
                        sequence = []
                        for path in method_representation['paths']:
                            sequence.append([value_and_path_vocab[path['valueLeft']], value_and_path_vocab[path['path']], value_and_path_vocab[path['valueRight']]])
                        if len(sequence) <= max_seq_length:
                            sequences.append(sequence)

    return sequences, labels


def generate_datapoints_write_tfrecord(path_to_dictionaries, path_to_dataset, max_seq_length, method_cutoff_value, num_records_file, file_path, file_name, train=True):
    """

    :param path_to_dictionaries:
    :param path_to_dataset:
    :param max_seq_length:
    :param num_records_file:
    :param file_path:
    :param file_name:
    :param train:
    :return: record_count:
    :return: drop_count:
    """
    if train:
        split_name = 'train'
    else:
        split_name = 'test'
    record_count, drop_count = 0, 0
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    value_and_path_vocab = {}
    with open(os.path.join(path_to_dictionaries, 'path_and_value_vocab.pkl'), 'rb') as file:
        value_and_path_vocab = pkl.load(file)
    # print(value_and_path_vocab)
    sequences = []
    labels = []
    for dirpath, dnames, fnames in os.walk(path_to_dataset):
        for f in fnames:
            if f.endswith(".json"):
                # The condition below ensures that it does not tries read a empty file
                if os.stat(os.path.join(dirpath, f)).st_size > 0:
                    json_data = open(os.path.join(dirpath, f), 'r')
                    json_data = json.load(json_data)
                    for method_representation in json_data:
                        if len(method_representation['paths']) <= max_seq_length:
                            labels.append(tag_vocab[method_representation['name']])
                            sequence = []
                            for path in method_representation['paths']:
                                sequence.append(
                                    [value_and_path_vocab[path['valueLeft']], value_and_path_vocab[path['path']],
                                     value_and_path_vocab[path['valueRight']]])
                            sequences.append(sequence)
                        else:
                            drop_count += 1
                    if len(sequences) > num_records_file:
                        fn_call_file_name = '%s_%s_%05d-of-n.tfrecord' % (file_name, split_name, record_count)
                        write_tfRecord(sequences[:num_records_file], labels[:num_records_file], os.path.join(file_path, fn_call_file_name))
                        print('Wrote TFRecord to ', os.path.join(file_path, fn_call_file_name))
                        record_count += 1
                        sequences, labels = sequences[num_records_file:], labels[num_records_file:]
    if len(sequences)> 0:
        fn_call_file_name = '%s_%s_%05d-of-n.tfrecord' % (file_name, split_name, record_count)
        write_tfRecord(sequences[:num_records_file], labels[:num_records_file],
                       os.path.join(file_path, fn_call_file_name))
        print('Wrote TFRecord to ', os.path.join(file_path, fn_call_file_name))
    return record_count, drop_count


def write_all_tfrecords(sequences, labels, num_records_file, file_path, file_name, train=True):
    split_name = ''
    if train:
        split_name = 'train'
    else:
        split_name = 'test'
    if len(sequences) % num_records_file == 0:
        _NUM_SHARDS = int(len(sequences)/num_records_file)
    else:
        _NUM_SHARDS = int((len(sequences)/num_records_file) + 1)
    for i in range(1, _NUM_SHARDS):
        fn_call_file_name = '%s_%s_%05d-of-%05d.tfrecord' % (file_name, split_name, i, _NUM_SHARDS)
        write_tfRecord(sequences[((i-1)*num_records_file):(i*num_records_file)], labels[((i-1) * num_records_file):(i*num_records_file)], os.path.join(file_path, fn_call_file_name))
        print('Wrote TFRecord to ', os.path.join(file_path, fn_call_file_name))
    if len(sequences) % num_records_file != 0:
        fn_call_file_name = '%s_%s_%05d-of-%05d.tfrecord' % (file_name, split_name, _NUM_SHARDS, _NUM_SHARDS)
        write_tfRecord(sequences[((_NUM_SHARDS-1)*num_records_file):len(sequences)], labels[((_NUM_SHARDS-1)*num_records_file):len(labels)], os.path.join(file_path, fn_call_file_name))

def plot_len(len_list):
    top = [(x, len_list.count(x)) for x in set(len_list)]
    print(top)
    # labels, ys = zip(*top)
    # xs = np.arange(len(labels))
    # width = 1
    #
    # plt.bar(xs, ys, width, align='center')
    #
    # plt.xticks(xs, labels)  # Replace default x-ticks with xs, then replace xs with labels
    # plt.yticks(ys)
    #
    # plt.savefig('/home/weave/Documents/code2vec_data/test_dataset/netscore.png')


def get_dict_lengths(path_to_dictionaries):
    value_path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_and_value_vocab.pkl'), 'rb'))
    path_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'path_vocab.pkl'), 'rb'))
    value_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'value_vocab.pkl'), 'rb'))
    tag_vocab = pkl.load(open(os.path.join(path_to_dictionaries, 'tag_vocab.pkl'), 'rb'))
    return len(value_vocab), len(path_vocab), len(tag_vocab), len(value_path_vocab)

def pickle_data(sequences, labels, path_to_pickle_data):
    pkl.dump(sequences, open(os.path.join(path_to_pickle_data, 'sequences.pkl'), 'wb'))
    pkl.dump(labels, open(os.path.join(path_to_pickle_data, 'labels.pkl'), 'wb'))

def unpickle_data(path_to_pickle_data):
    sequences = pkl.load(open(os.path.join(path_to_pickle_data, 'sequences.pkl'), 'rb'))
    labels = pkl.load(open(os.path.join(path_to_pickle_data, 'labels.pkl'), 'rb'))
    return sequences, labels


path_to_data = '/home/weave/Documents/code2vec_data/github_dataset/java_representation_ast/'
path_to_dic = '/home/weave/Documents/code2vec_data/github_dataset/dictionaries/'
path_to_tfrecord  = '/home/weave/Documents/code2vec_data/github_dataset/tfrecord_files/'
max_seq_len, records_per_file = 5000, 20000
# generate_datapoints_write_tfrecord(path_to_dic, path_to_data, max_seq_len, records_per_file, path_to_tfrecord, file_name ='code2vec_data')
print(get_dict_lengths(path_to_dic))



#####################################################################################################################################################
# Stats about the dataset
#####################################################################################################################################################
# Number of projects:
# Number of Java Files: 1908323
# Number of Methods aka datapoints: 1,75,40,000 (approx)
#

#####################################################################################################################################################
# path_to_data = '/home/weave/Documents/code2vec_data/test_dataset/java_representation_ast/'
# path_to_dic = '/home/weave/Documents/code2vec_data/test_dataset/dictionaries/'
# path_to_tfrecord  = '/home/weave/Documents/code2vec_data/test_dataset/tfrecord_files/'
# path_to_pickle_data = '/home/weave/Documents/code2vec_data/test_dataset/pickled_data'
# max_seq_length = 5000
# sequences, labels = get_datapoints_combined(path_to_dic,path_to_data,max_seq_length)
# pickle_data(sequences, labels, path_to_pickle_data)
# write_all_tfrecords(sequences=sequences, labels=labels, num_records_file=5000, file_path=path_to_tfrecord, file_name ='code2vec_data')
# count = 0
# print(len(sequences))
# for sequence in sequences:
#     if len(sequence)> 5000:
#         count+=1
# print(count)
#####################################################################################################################################################

# path_to_dic = '/Users/amankhandelia/Documents/machine_learning/code2vec/data/'
# print(get_dict_lengths(path_to_dic))
# ###########################################################################################
# path_to_tfrecord  = '/Users/amankhandelia/Documents/machine_learning/code2vec/tfrecord_files/'
# write_all_tfrecords(sequences, labels, 500, file_path=path_to_tfrecord, file_name='code2vec_data')
# ###########################################################################################
# sequences, labels = get_datapoints(path_to_dic, path_to_dic)
# ###########################################################################################
# path_to_tfrecord  = '/Users/amankhandelia/Documents/machine_learning/code2vec/tfrecord_files/train.tfRecord'
# dataset = tf.data.TFRecordDataset(path)
# dataset = dataset.map(parse_sequence)
# dataset = dataset.map(expand).padded_batch(3, padded_shapes=({'length':1}, {'sequence':tf.TensorShape([None, 3]), 'labels':1})).map(deflate)
# iterator = dataset.make_one_shot_iterator()
# features = iterator.get_next()
# sess = tf.Session()
# data = sess.run(features)
# print(data)
# ###########################################################################################
# path_to_data = '/Users/amankhandelia/Documents/machine_learning/code2vec/data/java_representation_ast/'
# path_to_dic = '/Users/amankhandelia/Documents/machine_learning/code2vec/data/'
# generate_and_save_dict(path_to_data, path_to_dic)
# write_tfRecord(sequences[:5], labels[:5], file_path=path_to_tfrecord)