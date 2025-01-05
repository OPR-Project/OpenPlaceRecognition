SequencePointcloudRegistrationPipeline
=======================================

A module that implements an algorithm for optimizing the position and orientation of a vehicle in space
based on a sequence of multimodal data using neural network methods.


Usage example
-------------

You should start with initializing neural model
:class:`opr.models.registration.hregnet.HRegNet`
with the desired configuration and weights.
For example, you can use the
`configs/model/registration/hregnet_light_feats.yaml <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/configs/model/registration/hregnet_light_feats.yaml>`_
config:

.. code-block:: python

   REG_MODEL_CONFIG_PATH = "configs/model/registration/hregnet_light_feats.yaml"
   REG_WEIGHTS_PATH = "weights/registration/hregnet_light_feats_nuscenes.pth"

   reg_model_config = OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH)
   reg_model = instantiate(reg_model_config)
   reg_model.load_state_dict(torch.load(REGISTRATION_WEIGHTS_PATH))

Then you should initialize the
:class:`opr.pipelines.registration.pointcloud.SequencePointcloudRegistrationPipeline`.

.. code-block:: python

   from opr.pipelines.registration import SequencePointcloudRegistrationPipeline

   DEVICE = "cuda"

   reg_pipe = PointcloudRegistrationPipeline(
       model=reg_model,
       model_weights_path=REG_WEIGHTS_PATH,
       device=DEVICE,
       voxel_downsample_size=0.3,
       num_points_downsample=8192,
   )

Then you can use the pipeline to infer transformation between two point clouds:

.. code-block:: python

   query_pc_1 = ...  # coordinates torch.Tensor of shape (N_1, 3)
   query_pc_2 = ...  # coordinates torch.Tensor of shape (N_2, 3)
   query_list = [query_pc_1, query_pc_2]

   db_pc = ...  # coordinates torch.Tensor of shape (M, 3)

   transform_matrix = loc_pipe.infer(query_pc_list=query_list, db_pc=db_pc)

Alternatively, you can use precomputed features for database point clouds:

.. code-block:: python

   query_pc_1 = ...  # coordinates torch.Tensor of shape (N_1, 3)
   query_pc_2 = ...  # coordinates torch.Tensor of shape (N_2, 3)
   query_list = [query_pc_1, query_pc_2]

   db_pc = ...  # coordinates torch.Tensor of shape (M, 3)
   db_pc_features = reg_model.extract_features(db_pc)

   transform_matrix = loc_pipe.infer(query_pc_list=query_list, db_pc_feats=db_pc_features)

The pipeline will return the 4x4 `np.ndarray` transformation matrix between the query and database point clouds.

More usage examples can be found in the following notebooks:

* `notebooks/test_cross_season/01_PlaceRecognitionPipeline.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_cross_season/01_PlaceRecognitionPipeline.ipynb>`_
* `notebooks/test_public_datasets_v2/02_SequencePointcloudRegistrationPipeline.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/notebooks/test_public_datasets_v2/02_SequencePointcloudRegistrationPipeline.ipynb>`_
