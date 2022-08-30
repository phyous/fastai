import torch
from fastai.vision.all import *

print(f"CUDA status: {torch.cuda.is_available()}")

path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
print(f"num files {len(files)}")
def label_func(f): return f[0].isupper()
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)