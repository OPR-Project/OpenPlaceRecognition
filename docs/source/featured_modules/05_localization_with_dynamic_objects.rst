LocalizationPipeline with dynamic objects
========================

To increase quality of place recognition with localization in highly dynamic environments you can consider to add objects masking
(both on lidar and images) to prevent feature extraction in those regions.


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
:class:`opr.pipelines.place_recognition.base.PlaceRecognitionPipeline`
and
:class:`opr.pipelines.registration.pointcloud.PointcloudRegistrationPipeline`.

.. code-block:: python

   from opr.pipelines.place_recognition import PlaceRecognitionPipeline
   from opr.pipelines.registration import PointcloudRegistrationPipeline
   from opr.pipelines.localization import LocalizationPipeline

   DATABASE_DIR = "/path/to/database"
   DEVICE = "cuda"

   pr_pipe = PlaceRecognitionPipeline(
       database_dir=DATABASE_DIR,
       model=pr_model,
       model_weights_path=PR_WEIGHTS_PATH,
       device=DEVICE,
   )
   reg_pipe = PointcloudRegistrationPipeline(
       model=reg_model,
       model_weights_path=REG_WEIGHTS_PATH,
       device=DEVICE,
       voxel_downsample_size=0.3,
       num_points_downsample=8192,
   )
   loc_pipe = LocalizationPipeline(
       place_recognition_pipeline=pr_pipe,
       registration_pipeline=reg_pipe,
       precomputed_reg_feats=True,
       pointclouds_subdir="lidar",
   )

To apply masking you should extract segmentation results and place them in data folder.

.. code-block:: python

    dataset = ITLPCampus(
        dataset_root=DATASET_ROOT,
        subset="test",
        csv_file="test.csv",
        sensors=SENSOR_SUITE,
        load_semantics=True,
        exclude_dynamic_classes=True,
        image_transform=ToTensorTransform(),
        semantic_transform=ToTensorTransform(),
        late_image_transform=DefaultImageTransform(resize=(320, 192), train=False)
    )

During getitem image and lidar data will be masked automatically by selected indices.
For proper lidar masking external transformation between sensors should be set in `opr.datasets.base.BasePlaceRecognitionDataset`.
You can find example in `opr.datasets.itlp.ITLPCampus`.

Then you can use the pipeline to infer the location of the input query data as a usual 
:class:`opr.pipelines.localization.base.LocalizationPipeline`
:

.. code-block:: python

   query_data = {
       "image_front": image_front,
       "image_back": image_back,
       "pointcloud_lidar_coords": pointcloud_lidar_coords,
       "pointcloud_lidar_feats": pointcloud_lidar_feats,
   }

   loc_pipe.infer(query_data)

The pipeline will return the output dictionary with the following keys:

* ``"db_match_pose"``: the pose of the most similar record in the database
* ``"estimated_pose"``: the estimated pose of the query data after registration

More usage examples can be found in the following notebooks:

* `notebooks/test_itlp/05_LocalizationPipeline_with_dynamic.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_itlp/05_LocalizationPipeline_with_dynamic.ipynb>`_
* `notebooks/test_cross_season/05_LocalizationPipeline_with_dynamic.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_cross_season/05_LocalizationPipeline_with_dynamic.ipynb>`_
