import torch

from opr.modules.feature_extractors.resnet import ResNet18FPNFeatureExtractor


WEIGHTS_PATH = "/home/docker_opr/OpenPlaceRecognition/weights/place_recognition/multi-image_multi-semantic_lidar_late-fusion_nclt.pth"
state_dict = torch.load(WEIGHTS_PATH)
image_state_dict = {k.removeprefix("image_module.backbone."):v for k, v in state_dict.items()
    if k.startswith("image_module.backbone.")}

device = torch.device('cuda:0')
model = ResNet18FPNFeatureExtractor()
model.load_state_dict(image_state_dict)
model.to(device)
model.eval();

input_names = ['input']
output_names = ['output']
dummy_input = (torch.randn(1, 3, 192, 320, device=device))
torch.onnx.export(model,
            dummy_input,
            "/home/docker_opr/OpenPlaceRecognition/weights/place_recognition/ResNet18FPN_ImageFeatureExtractor.onnx",
            verbose=True,
            opset_version=13,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={'input' : {0 : 'batch_size'},
                          'output' : {0 : 'batch_size'}},
            export_params=True,
            do_constant_folding=True,
)
