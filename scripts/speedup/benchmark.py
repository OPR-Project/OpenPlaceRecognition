from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import torch_tensorrt
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})


def benchmark(model, sample, forward_type):
    with torch.no_grad():
        print(f'{model.__class__.__name__} {forward_type} inference mode:')
        print()

        latency = []
        for _ in range(5):
            model(sample)
        for _ in range(500):
            t0 = time.time()
            model(sample)
            torch.cuda.synchronize()
            latency.append(1000 * (time.time() - t0))

        print(f"{np.mean(latency):.2f} ms +/- {np.std(latency):.2f} ms")
        print('==========================')

def benchmark_fp16(model, sample):
    with torch.cuda.amp.autocast():
        benchmark(model, sample, "fp16")


if __name__ == "__main__":
    MODEL_CONFIG_PATH = "configs/model/place_recognition/speedup_benchmark.yaml"
    DEVICE = "cuda"

    model_config = OmegaConf.load(MODEL_CONFIG_PATH)
    model = instantiate(model_config)
    model = model.to(DEVICE)
    model.eval();

    batch = {"images_": torch.rand(1, 3, 192, 320, device=DEVICE),
             "masks_": torch.rand(1, 1, 192, 320, device=DEVICE),
             "soc": torch.rand(1, 72, 10, 3, device=DEVICE)}

    if model.image_module:
        benchmark(model.image_module, batch,
                  forward_type=model.image_module.forward_type)
        #benchmark_fp16(model.image_module, batch)

    if model.semantic_module:
        benchmark(model.semantic_module, batch,
                  forward_type=model.semantic_module.forward_type)
    
    if model.cloud_module:
        print("Skip - MinkowskiEngine")

    if model.soc_module:
        benchmark(model.soc_module, batch,
                  forward_type=model.semantic_module.forward_type)
