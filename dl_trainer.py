import fastbook
from fastbook import *
from fastai.vision.widgets import *
import pickle
import os

def return_model(model,data):
    if model == 1:
        learn = cnn_learner(data, resnet18, metrics=[error_rate, accuracy])
    elif model == 2:
        learn = cnn_learner(data, resnet34, metrics=[error_rate, accuracy])
    elif model == 3:
        learn = cnn_learner(data, resnet50, metrics=[error_rate, accuracy])
    elif model == 4:
        learn = cnn_learner(data, resnet101, metrics=[error_rate, accuracy])
    elif model == 5:
        learn = cnn_learner(data, resnet152, metrics=[error_rate, accuracy])
    elif model == 6:
        learn = cnn_learner(data, densenet121, metrics=[error_rate, accuracy])
    elif model == 7:
        learn = cnn_learner(data, densenet169, metrics=[error_rate, accuracy])
    elif model == 8:
        learn = cnn_learner(data, densenet201, metrics=[error_rate, accuracy])
    elif model == 9:
        learn = cnn_learner(data, densenet161, metrics=[error_rate, accuracy])
    elif model == 10:
        learn = cnn_learner(data, alexnet, metrics=[error_rate, accuracy])
    elif model == 11:
        learn = cnn_learner(data, squeezenet1_0, metrics=[error_rate, accuracy])
    elif model == 12:
        learn = cnn_learner(data, squeezenet1_1, metrics=[error_rate, accuracy])
    elif model == 13:
        learn = cnn_learner(data, vgg11, metrics=[error_rate, accuracy])
    elif model == 14:
        learn = cnn_learner(data, vgg11_bn, metrics=[error_rate, accuracy])
    elif model == 15:
        learn = cnn_learner(data, vgg13, metrics=[error_rate, accuracy])
    elif model == 16:
        learn = cnn_learner(data, vgg13_bn, metrics=[error_rate, accuracy])
    elif model == 17:
        learn = cnn_learner(data, vgg16, metrics=[error_rate, accuracy])
    elif model == 18:
        learn = cnn_learner(data, vgg16_bn, metrics=[error_rate, accuracy])
    elif model == 19:
        learn = cnn_learner(data, vgg19, metrics=[error_rate, accuracy])
    elif model == 20:
        learn = cnn_learner(data, vgg19_bn, metrics=[error_rate, accuracy])
    return learn


def dl_models(filename,model=1,aug=1,epochs=4,validation_split=0.2):
    try:
        os.mkdir("models")
    except:
        pass
    if (aug==1):
        item_tfms1 = Resize(128, ResizeMethod.Squish)
        item_tfms2 = Resize(128, ResizeMethod.Pad, pad_mode='zeros')
        item_tfms3 = RandomResizedCrop(128, min_scale=0.3)
        item_tfms = [item_tfms3,item_tfms2,item_tfms1]
        tfms = aug_transforms(do_flip=True, flip_vert=False, mult=2.0)
        data = ImageDataLoaders.from_folder(filename, valid_pct=validation_split, item_tfms=item_tfms, batch_tfms=tfms, bs=30,
                                            num_workers=4)
        learn = return_model(model,data)
        learn.fine_tune(epochs)
        preds, targs = learn.tta()
        print("Current Directory : ",os.getcwd())
        learn.export('/content/models/model.pkl')
        print("model trained.saving model..")
        filename_model = '/content/models/model.pkl'
        return filename_model,(accuracy(preds, targs).item())
    else:
        data = ImageDataLoaders.from_folder(filename, valid_pct=validation_split)
        learn = return_model(model, data)
        learn.fine_tune(epochs)
        preds, targs = learn.tta()
        learn.export('./models/model.pkl')
        print("model trained.saving model..")
        filename_model = './models/model.pkl'
        return (filename_model, 100*(accuracy(preds, targs).item()))