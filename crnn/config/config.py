GPU_ID = None
GPU_AMOUNT = 1

model_path = '/home/rid/PythonProjects/TEST/Table-OCR/crnn/checkpoints/best_model.pth'
CNN = {'MODEL': "ResNet18"}
RNN = {'MODEL': "lstm_2layer",
       'multi_gpu': False,
       'n_In': 512,
       'n_Hidden': 256,
       'n_Layer': 2,
       'dropout': 0.5}
