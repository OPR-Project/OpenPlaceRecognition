.. opr documentation master file, created by
   sphinx-quickstart on Mon Dec 23 16:22:10 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##############################
Open Place Recognition library
##############################

.. image:: ../images/PR_overview.png

*An overview of a typical place recognition pipeline.
At first, the input data is encoded into a query descriptor.
Then, a K-nearest neighbors search is performed between the query and the database.
Finally, the position of the closest database descriptor found is considered as the answer.*

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Featured modules
================

*WIP*


Installation
============

Pre-requisites
--------------

* The library requires `PyTorch`, `MinkowskiEngine` and (optionally) `faiss` libraries
  to be installed manually:

  * `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_
  * `MinkowskiEngine repository <https://github.com/NVIDIA/MinkowskiEngine>`_
  * `faiss repository <https://github.com/facebookresearch/faiss>`_

* Another option is to use the docker image.
  Quick-start commands to build, start and enter the container:

  .. code-block:: bash

     # from repo root dir
     bash docker/build_devel.sh
     bash docker/start.sh [DATASETS_DIR]
     bash docker/into.sh


Library installation
--------------------

After the pre-requisites are met, install the Open Place Recognition library with the following command:

.. code-block:: bash

   pip install -e .


Third-party packages
--------------------

* If you want to use the `GeoTransformer` model for pointcloud registration,
  you should install the package located in the `third_party` directory:

  .. code-block:: bash

     # load submodules from git
     git submodule update --init

     # change dir
     cd third_party/GeoTransformer/

     # install the package
     bash setup.sh

* If you want to use the `HRegNet` model for pointcloud registration,
  you should install the package located in the `third_party` directory:

  .. code-block:: bash

     # load submodules from git
     git submodule update --init

     # change dir
     cd third_party/HRegNet/

     # at first, install PointUtils dependency
     python hregnet/PointUtils/setup.py install

     # then, install the package hregnet
     pip install .



How to load the weights
-----------------------

You can download the weights from the public
`Google Drive folder <https://drive.google.com/drive/folders/1uRiMe2-I9b5Tgv8mIJLkdGaHXccp_UFJ?usp=sharing>`_.


ITLP-Campus dataset
===================

We introduce the ITLP-Campus dataset.
The dataset was recorded on the Husky wheeled robotic platform on the university campus
and consists of tracks recorded at different times of day (day/dusk/night) and different seasons (winter/spring).
You can find more details in the `OPR-Project/ITLP-Campus <https://github.com/OPR-Project/ITLP-Campus>`_ repository.


Package Structure
=================

opr.datasets
------------

Subpackage containing dataset classes and functions.

Usage example:

.. code-block:: python

   from opr.datasets import OxfordDataset

   train_dataset = OxfordDataset(
       dataset_root="/home/docker_opr/Datasets/pnvlad_oxford_robotcar_full/",
       subset="train",
       data_to_load=["image_stereo_centre", "pointcloud_lidar"]
   )

The iterator will return a dictionary with the following keys:

* ``"idx"``: index of the sample in the dataset, single number Tensor
* ``"utm"``: UTM coordinates of the sample, Tensor of shape ``(2)``
* (optional) ``"image_stereo_centre"``: image Tensor of shape ``(C, H, W)``
* (optional) ``"pointcloud_lidar_feats"``: point cloud features Tensor of shape ``(N, 1)``
* (optional) ``"pointcloud_lidar_coords"``: point cloud coordinates Tensor of shape ``(N, 3)``

More details can be found in the `demo_datasets.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/demo_datasets.ipynb>`_ notebook.


opr.losses
----------

The ``opr.losses`` subpackage contains ready-to-use loss functions implemented in PyTorch, featuring a common interface.

Usage example:

.. code-block:: python

   from opr.losses import BatchHardTripletMarginLoss

   loss_fn = BatchHardTripletMarginLoss(margin=0.2)

   idxs = sample_batch["idxs"]
   positives_mask = dataset.positives_mask[idxs][:, idxs]
   negatives_mask = dataset.negatives_mask[idxs][:, idxs]

   loss, stats = loss_fn(output["final_descriptor"], positives_mask, negatives_mask)

The loss functions introduce a unified interface:

* **Input:**

  * ``embeddings``: descriptor Tensor of shape ``(B, D)``
  * ``positives_mask``: boolean mask Tensor of shape ``(B, B)``
  * ``negatives_mask``: boolean mask Tensor of shape ``(B, B)``

* **Output:**

  * ``loss``: loss value Tensor
  * ``stats``: dictionary with additional statistics

More details can be found in the `demo_losses.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/demo_losses.ipynb>`_ notebook.


opr.models
----------

The ``opr.models`` subpackage contains ready-to-use neural networks implemented in PyTorch, featuring a common interface.

Usage example:

.. code-block:: python

   from opr.models.place_recognition import MinkLoc3D

   model = MinkLoc3D()

   # forward pass
   output = model(batch)

The models introduce unified input and output formats:

* **Input:** a ``batch`` dictionary with the following keys
  (all keys are optional, depending on the model and dataset):

  * ``"images_<camera_name>"``: images Tensor of shape ``(B, 3, H, W)``
  * ``"masks_<camera_name>"``: semantic segmentation masks Tensor of shape ``(B, 1, H, W)``
  * ``"pointclouds_lidar_coords"``: point cloud coordinates Tensor of shape ``(B * N_points, 4)``
  * ``"pointclouds_lidar_feats"``: point cloud features Tensor of shape ``(B * N_points, C)``

* **Output:** a dictionary with the requiered key ``"final_descriptor"``
  and optional keys for intermediate descriptors:

  * ``"final_descriptor"``: final descriptor Tensor of shape ``(B, D)``

More details can be found in the `demo_models.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/demo_models.ipynb>`_ notebook.


opr.trainers
------------

The ``opr.trainers`` subpackage contains ready-to-use training algorithms.

Usage example:

.. code-block:: python

   from opr.trainers.place_recognition import UnimodalPlaceRecognitionTrainer

   trainer = UnimodalPlaceRecognitionTrainer(
       checkpoints_dir=checkpoints_dir,
       model=model,
       loss_fn=loss_fn,
       optimizer=optimizer,
       scheduler=scheduler,
       batch_expansion_threshold=cfg.batch_expansion_threshold,
       wandb_log=(not cfg.debug and not cfg.wandb.disabled),
       device=cfg.device,
   )

   trainer.train(
       epochs=cfg.epochs,
       train_dataloader=dataloaders["train"],
       val_dataloader=dataloaders["val"],
       test_dataloader=dataloaders["test"],
   )


opr.pipelines
--------------

The ``opr.pipelines`` subpackage contains ready-to-use pipelines for model inference.

Usage example:

.. code-block:: python

   from opr.models.place_recognition import MinkLoc3Dv2
   from opr.pipelines.place_recognition import PlaceRecognitionPipeline

   pipe = PlaceRecognitionPipeline(
       database_dir="/home/docker_opr/Datasets/ITLP_Campus/ITLP_Campus_outdoor/databases/00",
       model=MinkLoc3Dv2(),
       model_weights_path=None,
       device="cuda",
   )

   out = pipe.infer(sample)

The pipeline introduces a unified interface for model inference:

* **Input:** a dictionary with the following keys
  (all keys are optional, depending on the model and dataset):

  * ``"image_<camera_name>"``: image Tensor of shape ``(3, H, W)``
  * ``"mask_<camera_name>"``: semantic segmentation mask Tensor of shape ``(1, H, W)``
  * ``"pointcloud_lidar_coords"``: point cloud coordinates Tensor of shape ``(N_points, 4)``
  * ``"pointcloud_lidar_feats"``: point cloud features Tensor of shape ``(N_points, C)``

* **Output:** a dictionary with keys:

  * ``"idx"`` for predicted index in the database,
  * ``"pose"`` for predicted pose in the format ``[tx, ty, tz, qx, qy, qz, qw]``,
  * ``"descriptor"`` for predicted descriptor.

More details can be found in the `demo_pipelines.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/demo_pipelines.ipynb>`_ notebook.


Model Zoo
=========

*WIP*


Connected Projects
==================

* `OPR-Project/OpenPlaceRecognition-ROS2 <https://github.com/OPR-Project/OpenPlaceRecognition-ROS2>`_ -
  ROS-2 implementation of OpenPlaceRecognition modules
* `OPR-Project/ITLP-Campus <https://github.com/OPR-Project/ITLP-Campus>`_ -
  ITLP-Campus dataset tools.
* `KirillMouraviev/simple_toposlam_model <https://github.com/KirillMouraviev/simple_toposlam_model>`_ -
  An implementation of the Topological SLAM method that uses the OPR library.


License
=======

**the license is subject to change in future versions**

.. include:: ../LICENSE
