import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

import crnn.utils.keys as keys
import crnn.utils.util as util
import crnn.models.crnn as crnn

from crnn.config import config


alphabet = keys.alphabet
nClass = len(alphabet) + 1


class Crnn_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id
        self.converter = util.strLabelConverter(alphabet)
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            checkpoint = torch.load(model_path)
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.device = torch.device("cpu")
        print('text recognition running on device:', self.device)

        self.net = crnn.CRNN(config, nClass)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()
        self.transform = transforms.Compose([transforms.ToTensor()])

    def predict(self, img):
        """
        :param img: Image object: L mode
        :return: char_list: top5 results of character recognition
        prob_list: top5 confidence
        char_position: single character region of the text box
        """
        copy_img = img.copy()
        img = img.point(lambda i: 255 - i)
        width, height = img.size[0], img.size[1]
        scale = height * 1.0 / 32
        width = int(width / scale)

        img = img.resize([width, 32], Image.BILINEAR)

        tensor = self.transform(img)
        tensor.sub_(0.5).div_(0.5)
        tensor = tensor.to(self.device)
        tensor = tensor.unsqueeze(0)

        predict = self.net(tensor)
        predict_len = Variable(torch.IntTensor([predict.size(0)] * 1))
        acc = predict.softmax(2).topk(5)
        char_list, prob_list, char_positions = self.converter.decode(acc, predict_len.data, copy_img, scale, raw=False)
        return char_list, prob_list, char_positions