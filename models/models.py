import torch
import torchvision.models as models
from torchvision import transforms


def load_model_from_name(model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading to: {device}')
    if model_name == 'inceptionv3':
        model = models.inception_v3(pretrained=True)
        model.to(device)
        model.eval()
        target_image_size = (299, 299)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.to(device)
        model.eval()
        target_image_size = (224, 224)
    else:
        model = None
        target_image_size = None
    return model, target_image_size


def get_preprocess_pipeline_for_model(target_image_size):
    preprocess = transforms.Compose([
        transforms.Resize(target_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess


def get_preprocess_pipeline_for_model_mscoco_version(target_image_size):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(target_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess

