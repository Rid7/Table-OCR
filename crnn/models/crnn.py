import crnn.models.convnet as ConvNets
import crnn.models.recurrent as SeqNets
import torch.nn as nn
import torch.nn.parallel


class CRNN(nn.Module):
    def __init__(self, config, n_class):
        super(CRNN, self).__init__()
        self.ngpu = config.GPU_AMOUNT
        cnn_conf = config.CNN
        self.cnn = ConvNets.__dict__[cnn_conf['MODEL']]()

        rnn_conf = config.RNN
        print('Constructing {}'.format(rnn_conf['MODEL']))
        self.rnn = SeqNets.__dict__[rnn_conf['MODEL']](rnn_conf, n_class)

    def forward(self, input):
        c_feat = data_parallel(self.cnn, input, self.ngpu)

        b, c, h, w = c_feat.size()
        assert h == 1, "the height of the conv must be 1"

        c_feat = c_feat.squeeze(2)
        c_feat = c_feat.permute(2, 0, 1)

        output = data_parallel(self.rnn, c_feat, self.ngpu, dim=1)
        # print(output[0])
        return output


def data_parallel(model, input, ngpu, dim=0):
    if isinstance(input.data, torch.cuda.FloatTensor) and ngpu:
        output = nn.parallel.data_parallel(model, input, range(ngpu), dim=dim)
    else:
        output = model(input)
    return output
