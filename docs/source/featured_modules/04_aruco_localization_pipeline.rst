ArucoLocalizationPipeline
========================

A module that implements a combination of neural network algorithm for place recognition and localization using
sequences of data from lidars and cameras and classic Aruco detection using cameras.


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

Then you should define information about camera configuration and markers in dict-like structure.
Sensor to baselink (f. e. front2baselink) can be used to add additional transformation to pose coordinate system ([x, y, z, qx, qy, qz, qw]).

.. code-block:: python

    camera_metadata = {
        "front_cam_intrinsics": [[683.61, 0.0, 615.12],
                                [0.0, 683.61, 345.32],
                                [0.0, 0.0, 1.0]],
        "front_cam_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "front_cam2baselink": [-0.23, 0.06, 0.75, -0.5, 0.49, -0.5, 0.50],
        "back_cam_intrinsics": [[910.41, 0.0, 648.44],
                                [0.0, 910.41, 354.01],
                                [0.0, 0.0, 1.0]],
        "back_cam_distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        "back_cam2baselink": [-0.37, -0.04, 0.74, -0.49, -0.49, 0.50, 0.55]
    }

    aruco_metadata = {
        "aruco_type": cv2.aruco.DICT_4X4_250,
        "aruco_size": 0.2,
        "aruco_gt_pose_by_id": {
            0: [-23.76, 16.94, 1.51, 0.25, 0.65, 0.65, 0.29],
            2: [-8.81, -12.47, 1.75, 0.61, -0.28, -0.21, 0.73],
        }
    }

initialize the
:class:`opr.pipelines.localization.base.ArucoLocalizationPipeline`
which consists of two sub-pipelines:
:class:`opr.pipelines.place_recognition.base.PlaceRecognitionPipeline`
and
:class:`opr.pipelines.registration.pointcloud.PointcloudRegistrationPipeline`.

.. code-block:: python

   from opr.pipelines.place_recognition import PlaceRecognitionPipeline
   from opr.pipelines.registration import PointcloudRegistrationPipeline
   from opr.pipelines.localization import ArucoLocalizationPipeline

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
   aruco_pipe = ArucoLocalizationPipeline(
       place_recognition_pipeline=pr_pipe,
       registration_pipeline=reg_pipe,
       precomputed_reg_feats=True,
       pointclouds_subdir="lidar",
       aruco_metadata=aruco_metadata,
       camera_metadata=camera_metadata
   )

Then you can use the pipeline to infer the location of the input query data:

.. code-block:: python

   query_data = {
       "image_front": image_front,
       "image_back": image_back,
       "pointcloud_lidar_coords": pointcloud_lidar_coords,
       "pointcloud_lidar_feats": pointcloud_lidar_feats,
   }

   aruco_pipe.infer(query_data)

The pipeline will return the output dictionary with the following keys:

* ``"pose_by_aruco"``: the estimated pose by Aruco detection and GT marker info. None if no Aruco detected
* ``"pose_by_place_recognition"``: the estimated pose of the query data after registration. Called in case of no Aruco presents detected.

More usage examples can be found in the following notebooks:

* `notebooks/test_itlp/04_ArucoLocalizationPipeline.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_itlp/04_ArucoLocalizationPipeline.ipynb>`_
* `notebooks/test_cross_season/04_ArucoLocalizationPipeline.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_cross_season/04_ArucoLocalizationPipeline.ipynb>`_
