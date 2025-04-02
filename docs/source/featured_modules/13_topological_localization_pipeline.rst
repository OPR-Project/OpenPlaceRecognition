TopologicalLocalizationPipeline
========================

A module which performs twofold localization (place recognition and pose registration) leveraging the topological properties of the environment.
The module builds a topological graph from the input database and outputs pose and corresponding node in the graph for each query.

Usage example
-------------

First, you should create the database for localization from a pre-processed dataset. The recommended way to do it is the usage of `opr.datasets` API. 
An example of building a database is shown in the notebook `notebooks/build_database.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/build_database.ipynb>`.

After that, you should initialize the
:class:`opr.pipelines.localization.base.TopologicalLocalizationPipeline`
which consists of two sub-pipelines:
:class:`opr.pipelines.place_recognition.base.PlaceRecognitionPipeline`
and
:class:`opr.pipelines.registration.pointcloud.PointcloudRegistrationPipeline`.
As a place recognition model, you can use
:class:`opr.models.place_recognition.base.LateFusionModel`.
As a registration model, you can use the `GeoTransformer <https://github.com/alexmelekhin/GeoTransformer/tree/1e56f104ee88cb60734ad9344e28b6d64536c2e8>` neural network model:

.. code-block:: python

    DATABASE_DIR = "/path/to/database"
    DEVICE = "cuda"
    MODEL_CONFIG_PATH = "../configs/model/place_recognition/multi-image_lidar_late-fusion.yaml"
    WEIGHTS_PATH = "../weights/place_recognition/multi-image_lidar_late-fusion_itlp-finetune.pth"

    pr_pipe = PlaceRecognitionPipeline(
        database_dir=DATABASE_DIR,
        model=model,
        model_weights_path=WEIGHTS_PATH,
        device=DEVICE,
    )

    REGISTRATION_MODEL_CONFIG_PATH = "../configs/model/registration/geotransformer_kitti.yaml"
    REGISTRATION_WEIGHTS_PATH = "../weights/registration/geotransformer_kitti.pth"

    geotransformer = instantiate(OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH))

    registration_pipe = PointcloudRegistrationPipeline(
        model=geotransformer,
        model_weights_path=REGISTRATION_WEIGHTS_PATH,
        device="cuda",  # the GeoTransformer currently only supports CUDA
        voxel_downsample_size=0.3,  # recommended for geotransformer_kitti configuration
    )

    topo_pipe = TopologicalLocalizationPipeline(
        database=database,
        place_recognition_pipeline=pr_pipe,
        registration_pipeline=registration_pipe,
        pointclouds_subdir='lidar',
        camera_names=['front_cam', 'back_cam'],
        multi_sensor_fusion=True,
        edge_threshold=5.0,
        top_k=5
    )

Then you can use the pipeline to infer the location of the input query data:

.. code-block:: python

   query_data = {
       "image_front": image_front,
       "image_back": image_back,
       "pointcloud_lidar_coords": pointcloud_lidar_coords,
       "pointcloud_lidar_feats": pointcloud_lidar_feats,
   }

   topo_pipe.infer(query_data)

The pipeline will return the output dictionary with the following keys:

* ``"db_match_pose"``: the pose of the location in the database found by place recognition model with topological restrictions.
* ``"current_node"``: the index of the found location in the topological graph.
* ``"estimated_pose"``: the estimated pose refined by point cloud registration.

More usage examples can be found in the notebook `notebooks/topological_pipeline.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/topological_pipeline.ipynb>`_
