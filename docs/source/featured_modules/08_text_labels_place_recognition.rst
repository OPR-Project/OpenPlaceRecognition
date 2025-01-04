TextLabelsPlaceRecognition
========================

A module that implements a neural network algorithm for searching a database
of places already visited by a vehicle for the most similar records using data from lidars, cameras and text label detections.


Usage example
-------------

You should start with initializing neural model
:class:`opr.models.place_recognition.base.LateFusionModel`
with the image and cloud modules.
The recommended way to do this is to use the
`configs/model/place_recognition/multi-image_lidar_late-fusion.yaml <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/configs/model/place_recognition/multi-image_lidar_late-fusion.yaml>`_
config file to instantiate the model with Hydra and load the weights from the
``"weights/place_recognition/multi-image_lidar_late-fusion_nclt.pth"``
or other file.

.. code-block:: python

   from hydra.utils import instantiate
   from omegaconf import OmegaConf


   PR_MODEL_CONFIG_PATH = "configs/model/place_recognition/multi-image_lidar_late-fusion.yaml"
   PR_WEIGHTS_PATH = "weights/place_recognition/multi-image_lidar_late-fusion_nclt.pth"

   pr_model_config = OmegaConf.load(PR_MODEL_CONFIG_PATH)
   pr_model = instantiate(pr_model_config)
   pr_model.load_state_dict(torch.load(PR_WEIGHTS_PATH))

With pr_model you should create pr_pipeline
:class:`opr.pipelines.place_recognition.text_labels.TextLabelsPlaceRecognitionOCRPipeline`
and intialize text detection and recognition model
:class:`opr.models.ocr.paddle.PaddleOcrPipeline`
:

.. code-block:: python

    from opr.models.ocr.paddle import PaddleOcrPipeline
    from opr.pipelines.place_recognition.text_labels import TextLabelsPlaceRecognitionOCRPipeline

    pr_pipe = TextLabelsPlaceRecognitionOCRPipeline(
        db_labels_path=db_labels_path,
        database_dir=database_dir,
        model=pr_model,
        model_weights_path=model_weights_path,
        device=device,
    )
    ocr_model = PaddleOcrPipeline(model_config)
    pr_pipe.init_ocr_model(ocr_model)

In the similar manner you should initialize the registration model with the
`configs/model/registration/hregnet_light_feats.yaml <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/configs/model/registration/hregnet_light_feats.yaml>`_
config:

.. code-block:: python

   REG_MODEL_CONFIG_PATH = "configs/model/registration/hregnet_light_feats.yaml"
   REG_WEIGHTS_PATH = "weights/registration/hregnet_light_feats_nuscenes.pth"

   reg_model_config = OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH)
   reg_model = instantiate(reg_model_config)
   reg_model.load_state_dict(torch.load(REGISTRATION_WEIGHTS_PATH))

Then you should initialize the
:class:`opr.pipelines.localization.base.LocalizationPipeline`
which consists of two sub-pipelines:
:class:`opr.pipelines.place_recognition.text_labels.TextLabelsPlaceRecognitionOCRPipeline`
and
:class:`opr.pipelines.registration.pointcloud.PointcloudRegistrationPipeline`.

Then you can use the pipeline to infer the location of the input query data:

.. code-block:: python

   from opr.pipelines.localization import LocalizationPipeline

   loc_pipe = LocalizationPipeline(
       place_recognition_pipeline=pr_pipe,
       registration_pipeline=reg_pipe,
       precomputed_reg_feats=True,
       pointclouds_subdir="lidar",
   )

   query_data = {
       "image_front": image_front,
       "image_back": image_back,
       "pointcloud_lidar_coords": pointcloud_lidar_coords,
       "pointcloud_lidar_feats": pointcloud_lidar_feats,
   }

   loc_pipe.infer(query_data)

The pipeline will return the output dictionary with the following keys:

* ``"db_match_pose"``: the pose of the most similar record in the database
* ``"db_match_idx"``: the index of the most similar record in the database
* ``"estimated_pose"``: the estimated pose of the query data after registration

More usage examples can be found in the following notebooks:

* `notebooks/test_itlp/08_TextLabelsPlaceRecognition.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_itlp/08_TextLabelsPlaceRecognition.ipynb>`_
* `notebooks/test_cross_season/08_TextLabelsPlaceRecognition.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_cross_season/08_TextLabelsPlaceRecognition.ipynb>`_
