import sys
import hiddenlayer as hl
import torch

from model import Model

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("usage:\n python {} model_checkpoint".format(sys.argv[0]))
    model_checkpoint = sys.argv[1]
    checkpoint = torch.load(model_checkpoint)

    model = Model("mld", 20, in_channels=16, channels=16,
                  retrain=True).cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.eval()
    print("----------------------------")
    print("MODEL LOADED FROM CHECKPOINT")
    print("----------------------------")
    hl.build_graph(model, torch.zeros([1, 3, 224, 224])).save("dnn.png")
