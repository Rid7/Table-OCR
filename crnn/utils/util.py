import torch
import collections


class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + '~'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text, depth=0):
        """Support batch or single str."""
        if isinstance(text, str):
            for char in text:
                if self.alphabet.find(char) == -1:
                    print(char)
            text = [self.dict[char] for char in text]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)

        if depth:
            return text, len(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, copy_img, scale, raw=False):
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            prob_list = []
            point_list = []

            cut_index = 0
            for i in range(length):
                if t[1][i][0][0] != 0 and (not (i > 0 and t[1][i - 1][0][0] == t[1][i][0][0])):
                    if i == length - 1:
                        point_list.append([int((i-int((i - cut_index)/2)) * 4 * scale), int((i+1) * 4 * scale)])
                    elif t[1][i + 1][0][0] == t[1][i][0][0]:
                        point_list.append([int((i - int((i - cut_index)/2)) * 4 * scale), int((i + 3) * 4 * scale)])
                    else:
                        point_list.append([int((i-int((i - cut_index)/2)) * 4 * scale), int((i+2) * 4 * scale)])
                    # if t0[1][i + 1][0][0] == t[1][i][0][0]:
                    #
                    #     Image.fromarray(
                    #         np.array(copy_img)[:, int((i - int((i - cut_index)/2)) * 4 * scale): int((i + 3) * 4 * scale)]).save(
                    #         'found/{}.jpg'.format(i))
                    # else:
                    #     Image.fromarray(np.array(copy_img)[:, int((i-int((i - cut_index)/2)) * 4 * scale): int((i+2) * 4 * scale)]).save('found/{}.jpg'.format(i))
                    cut_index = i
                    char_list.append([self.alphabet[t[1][i][0][0] - 1], self.alphabet[t[1][i][0][1] - 1], self.alphabet[t[1][i][0][2] - 1], self.alphabet[t[1][i][0][3] - 1], self.alphabet[t[1][i][0][4] - 1]])
                    prob_list.append(t[0][i][0].tolist())
            return char_list, prob_list, point_list

    def decode_text(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            t = t[:length]
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(self.decode_text(
                    t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

