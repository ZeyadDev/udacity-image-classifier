import argparse
from util import load_model_checkpoint, predict
import torch
import json


def arg_parser():
    ap = argparse.ArgumentParser("Image Classifier prediction application")
    
    ap.add_argument("image_path", type=str, help="image path to predict")
    ap.add_argument("checkpoint", type=str, help="model checkpoint file path")
    ap.add_argument("--top_k", type=int, default=1, help="top k classes that model predicts")
    ap.add_argument("--category_names_file", type=str, help="file path to map categories to name")
    ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")

    return ap.parse_args()

if __name__ == "__main__":
    a = arg_parser()
    
    if a.category_names_file:
        file = open(a.category_names_file, 'r')
        categories = json.load(file)
    image_path = a.image_path
    checkpoint_file = a.checkpoint
    top_k = a.top_k
    gpu = a.gpu
    
    device = torch.device("cuda:0" if gpu else "cpu")

    device_type = "GPU" if gpu else "CPU"
    print(f"Use {device_type} for training")
   
    model,class_to_idx  = load_model_checkpoint(checkpoint_file, device)
    
    probs, classes = predict(image_path=image_path, model=model, device=device, topk=top_k, class_to_idx=class_to_idx)
    
    print("result:\n")
    
    probs = probs[0].tolist()
    
    for i in range(len(probs)):
        c = classes[i]
        c = categories[str(c)].capitalize()  if a.category_names_file else c
        print(f"Class {c} has probability of {probs[i]:.3f}" )