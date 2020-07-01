import numpy as np
import pickle

p_dict = pickle.load(open('pkl_file.pkl', 'rb'))


def viterbi_decode(score, transition_params):
    trellis = np.zeros_like(score)
    trellis[0] = score[0]
    backpointers = np.zeros_like(score, dtype=np.int32)
    for t in range(1, len(score)):
        matrix_node = np.expand_dims(trellis[t - 1], axis=1) + transition_params
        trellis[t] = score[t] + np.max(matrix_node, axis=0)

        backpointers[t] = np.argmax(matrix_node, axis=0)

    viterbi = [np.argmax(trellis[-1], axis=0)]
    for backpointer in reversed(backpointers[1:]):
        viterbi.append(backpointer[viterbi[-1]])
    viterbi_score = np.max(trellis[-1])
    viterbi.reverse()
    return viterbi, viterbi_score


def viterbi(nodes):
    # paths = nodes[0]
    paths = {i:0 for i in nodes[0].keys()}

    for l in range(1, len(nodes)):
        paths_ = paths.copy()
        paths = {}
        for i in nodes[l].keys():

            nows = {}
            for j in paths_.keys():
                try:
                    # nows[j + i] = paths_[j] + nodes[l][i] + p_dict[j[-1] + i] * 10
                    nows[j + i] = paths_[j] +  p_dict[j[-1] + i]
                except:
                    nows[j + i] = paths_[j] 
            k = np.argmax(list(nows.values()))
            paths[list(nows.keys())[k]] = list(nows.values())[k]
    return list(paths.keys())[np.argmax(list(paths.values()))]


def calculate(score):
    # score = np.array([[1, 2, 3],
    #           [2, 1, 3],
    #           [1, 3, 2],
    #           [3, 2,1]])  # (batch_size, time_step, num_tabs)
    # transition = np.array(len(labels),)# (num_tabs, num_tabs)
    if len(score[0]) == 1:
        if isinstance(score[0][0], str):
            return score[0][0]
        else:
            return score[0][0][0]
    elif score[1] == []:
        return ''.join(score[0])
    labels = []
    pred_sco = []
    j = 0
    out_str = ''
    list_len = len(score[1][0])
    for char in score[0]:
        try :
            if int(char[0]) in range(10) or ord(char[0]) in range(65,123):
                char = char[0]
        except Exception:
            pass
        if len(char) == 1:
            labels.append([char for i in range(list_len)])
            pred_sco.append([1 for i in range(list_len)])
        else:
            labels.append(char)
            pred_sco.append(score[1][j])
            j += 1

    nodes = list(map(lambda i: {labels[i][j]: pred_sco[i][j] for j in range(len(labels[0]))}, range(len(labels))))
    # transition = np.zeros((len(labels)-1, len(labels[0])))
    # a = transition.shape
    # for i in range(a[0]):
    #     for j in range(a[1]):
    #         ch = labels[i+1][i] + labels[1][j]
    #         transition[i][j] = p_dict[ch]
    # #score
    # print("=============")  # tensorflow
    # text = map(lambda i: score(pl[:, cut_position[i]:cut_position[i + 1] + 1], mode='search'),
    #            range(len(cut_position) - 1))

    tf_op = viterbi(nodes)
    return tf_op
