import torch
from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    engine_bytes_from_network,
    network_from_onnx_path,
)

from opr.datasets.itlp import ITLPCampus


def calib_data():
    for item in db_dataset_calib:
        yield {"input": item["image_front_cam"][None].contiguous().to("cuda")}

db_dataset_calib = ITLPCampus(
    dataset_root="/home/docker_opr/Datasets/ITLP-Campus-data/subsampled_data/indoor/00_2023-10-25-night",
    sensors=["front_cam"],
    mink_quantization_size=0.5,
    load_semantics=True,
    subset="test",
    test_split=[2]
)

calibrator = Calibrator(data_loader=calib_data())

serialized_engine = engine_bytes_from_network(
    network_from_onnx_path("/home/docker_opr/OpenPlaceRecognition/weights/place_recognition/ResNet18FPN_ImageFeatureExtractor.onnx"),
    config=CreateConfig(int8=True, calibrator=calibrator),
)

with open("ResNet18FPN_ImageFeatureExtractor.engine", "wb") as binary_file:
    binary_file.write(serialized_engine)
