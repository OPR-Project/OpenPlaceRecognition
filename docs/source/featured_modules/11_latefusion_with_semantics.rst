LateFusionModel with semantics
==============================

A module that implements a neural network algorithm for encoding records
using sequences of data from lidars and cameras.


Usage example
-------------

You should start with initializing neural model
:class:`opr.models.place_recognition.base.LateFusionModel`
with the image, semantic and cloud modules.
The recommended way to do this is to use the
`configs/model/place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/configs/model/place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml>`_
config file to instantiate the model with Hydra and load the weights from the
``"weights/place_recognition/multi-image_multi-semantic_lidar_late-fusion_nclt.pth"``
or other file.

.. code-block:: python

   from hydra.utils import instantiate
   from omegaconf import OmegaConf


   PR_MODEL_CONFIG_PATH = "configs/model/place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml"
   PR_WEIGHTS_PATH = "weights/place_recognition/multi-image_multi-semantic_lidar_late-fusion_nclt.pth"

   pr_model_config = OmegaConf.load(PR_MODEL_CONFIG_PATH)
   pr_model = instantiate(pr_model_config)
   pr_model.load_state_dict(torch.load(PR_WEIGHTS_PATH))

Then you can use the ``pr_model`` to infer the sensor's data:

.. code-block:: python

   query_data = {
       "image_front": image_front,
       "image_back": image_back,
       "mask_front": mask_front,
       "mask_back": mask_back,
       "pointcloud_lidar_coords": pointcloud_lidar_coords,
       "pointcloud_lidar_feats": pointcloud_lidar_feats,
   }

   output = pr_model(query_data)

The ``pr_model`` will return the ``output`` dictionary with the following keys:

* ``"final_descriptor"``: fused descriptor (torch.tensor) of all sensor's data
* ``"image"`` (optional): descriptor (torch.tensor) of image in sequence
* ``"semantic"`` (optional): descriptor (torch.tensor) of image semantic mask in sequence
* ``"cloud"`` (optional): descriptor (torch.tensor) of lidar point cloud in sequence

More usage examples of ``LateFusionModel`` can be found in the following notebooks as a part of ``pr_model``:

* `notebooks/test_itlp/09_LateFusionModel_with_semantics.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_itlp/09_LateFusionModel_with_semantics.ipynb>`_
* `notebooks/test_cross_season/09_LateFusionModel_with_semantics.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/test_cross_season/09_LateFusionModel_with_semantics.ipynb>`_
