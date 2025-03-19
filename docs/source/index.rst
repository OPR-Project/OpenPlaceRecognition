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

This library is suitable for:

* 🚗 **Navigation of autonomous cars, robots, and drones** using cameras and lidars, especially in areas with limited or unavailable GPS signals.
* 📦 **Localization of delivery robots** needing reliable positioning both indoors and outdoors.
* 🔬 **Research and development of computer vision algorithms**, related to multimodal place recognition and localization.
* 🎓 **Educational purposes and research projects**, involving robotics, autonomous systems, and computer vision.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Introduction <self>
   featured_modules/index
   itlp_dataset
   api/index


Installation
============

Quick-start
-----------

At first, ensure that you cloned the repository https://github.com/OPR-Project/OpenPlaceRecognition
and changed the directory to the root of the repository:

.. code-block:: bash

   git clone https://github.com/OPR-Project/OpenPlaceRecognition.git
   cd OpenPlaceRecognition

   # do not forget to load git submodules
   git submodule update --init

The recommended and easiest way to use the library is through the provided Docker environment.
The ``Dockerfile.base`` contains all prerequisites needed to run the library,
including optional `MinkowskiEngine <https://github.com/NVIDIA/MinkowskiEngine>`_ and `faiss <https://github.com/facebookresearch/faiss>`_ libraries.
You can either pull this image from Docker Hub (``docker pull alexmelekhin/open-place-recognition:base``),
or build it manually (``bash docker/build_base.sh``).

The ``Dockerfile.devel`` installs additional requirements from
``requirements.txt``,
``requirements-dev.txt``,
and ``requirements-notebook.txt`` files,
and creates a non-root user inside the image to avoid permission issues when using the container with mounted volumes.

The ``devel`` version of the image should be built manually by the user:

.. code-block:: bash

   bash docker/build_devel.sh

When starting the container, you must provide a data directory that will be mounted to ``~/Datasets`` inside the container:

.. code-block:: bash

   bash docker/start.sh [DATASETS_DIR]

To enter the container's ``/bin/bash`` terminal, use the ``docker/into.sh`` script:

.. code-block:: bash

   bash docker/into.sh

After you enter the container, install the library (we recommend installing in editable mode with the ``-e`` flag to be able to make modifications to the code):

.. code-block:: bash

   pip install -e ~/OpenPlaceRecognition


Third-party packages
--------------------

Some modules and pipelines require third-party packages to be installed manually. You can install these dependencies with the following commands:

* If you want to use the `GeoTransformer` model for pointcloud registration,
  you should install the package located in the `third_party` directory:

  .. code-block:: bash

     # load submodules from git
     git submodule update --init

     # change dir
     cd third_party/GeoTransformer/

     # install the package
     bash setup.sh

  If you are seeing `Permission denied` error, you should run the command with `sudo`:

  .. code-block:: bash

     # ... previous commands are the same

     sudo bash setup.sh

* If you want to use the `HRegNet` model for pointcloud registration,
  you should install the package located in the `third_party` directory:

  .. code-block:: bash

     # load submodules from git
     git submodule update --init

     # change dir
     cd third_party/HRegNet/

     # at first, install PointUtils dependency
     cd hregnet/PointUtils
     python setup.py install

     # then, install the package hregnet
     cd ../..  # go back to the third_party/HRegNet/ directory
     pip install .

  If you are seeing `Permission denied` error while trying to install PointUtils, you should run the command with `sudo`:

  .. code-block:: bash

     # ... previous commands are the same

     # install PointUtils dependency
     cd hregnet/PointUtils
     sudo python setup.py install

     # then, install the package hregnet
     cd ../..  # go back to the third_party/HRegNet/ directory
     sudo pip install .

**Note:** If you are using the provided Docker environment,
the default password for the user `docker_opr` can be found in the ``Dockerfile.devel`` file:

.. code-block:: bash

   # add user and his password
   ENV USER=docker_opr
   ARG UID=1000
   ARG GID=1000
   # default password
   ARG PW=user

You can change the password by providing the `PW` build-time argument in the `docker/build_devel.sh` script:

.. code-block:: bash

   docker build $PROJECT_ROOT_DIR \
       --build-arg PW=<new_password> \
       # other arguments


Advanced
--------

For a manual installation, you'll need to install several prerequisite libraries:

* **PyTorch** is the main dependency for our library. We recommend using version ``>=2.1.2``.
  Please refer to the `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ documentation for installation instructions.
* **torchvision** is a PyTorch library that provides datasets, transforms, and models for computer vision.
  It should be installed with PyTorch. Please refer to the `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ documentation.
  If you want to install it separately, please refer to the `torchvision GitHub repository <https://github.com/pytorch/vision>`_.
* **MinkowskiEngine** is a library for sparse tensor operations. We recommend using the fork of the library, which is compatible with CUDA 12.
  Please refer to the `Official Installation Guide <https://github.com/NVIDIA/MinkowskiEngine/wiki/Installation>`_,
  but instead of the official repository, use the fork: `<https://github.com/alexmelekhin/MinkowskiEngine.git>`_.
* **faiss** is a library for efficient similarity search and clustering of dense vectors.
  Please refer to the `faiss GitHub repository <https://github.com/facebookresearch/faiss>`_ for installation instructions.
  The recommended version is ``>=1.7.4``.
  We recommend using the GPU version of the library because our pipelines use it for better performance.
* **Open3D** is a library for 3D data processing.
  Please refer to the `Open3D documentation <https://www.open3d.org/docs/release/getting_started.html>`_ for installation instructions.
  We recommend using the GPU version of the library because our pipelines use it for better performance.
* **PaddlePaddle** and **PaddleOCR** are used for text-based place recognition pipelines.
  Please refer to the `PaddlePaddle Quick Start Guide <https://www.paddlepaddle.org.cn/en/install/quick>`_ for installation instructions.
  We recommend using the GPU version of the library because our pipelines use it for better performance.
  After installing PaddlePaddle, you can install PaddleOCR via pip: ``pip install paddleocr``.
* The library also depends on several performance optimization libraries that should be installed:

  * ONNX: refer to the `ONNX GitHub repository <https://github.com/onnx/onnx?tab=readme-ov-file#installation>`_.
  * ONNX Runtime: refer to the `Installation Guide <https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x>`_.
  * TensorRT: refer to the `Installation Guide <https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html#python-package-index-installation>`_.
  * Torch-TensorRT: refer to the `Installation Guide <https://pytorch.org/TensorRT/getting_started/installation.html>`_.
  * Polygraphy: refer to the `Polygraphy GitHub repository <https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy#installation>`_.

Please note that the manual installation of the library and its dependencies can be challenging and time-consuming.
We recommend using the provided Docker environment for a quick start.

If you encounter any issues during the installation process,
please refer to the `OpenPlaceRecognition Issues <https://github.com/OPR-Project/OpenPlaceRecognition/issues>`_ page.


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
       dataset_root="/home/docker_opr/Datasets/OpenPlaceRecognition/pnvlad_oxford_robotcar",
       subset="train",
       data_to_load=["image_stereo_centre", "pointcloud_lidar"]
   )

The iterator will return a dictionary with the following keys:

* ``"idx"``: index of the sample in the dataset, single number Tensor
* ``"utm"``: UTM coordinates of the sample, Tensor of shape ``(2)``
* (optional) ``"image_stereo_centre"``: image Tensor of shape ``(C, H, W)``
* (optional) ``"pointcloud_lidar_feats"``: point cloud features Tensor of shape ``(N, 1)``
* (optional) ``"pointcloud_lidar_coords"``: point cloud coordinates Tensor of shape ``(N, 3)``

In this example, we use a pre-processed version of the Oxford RobotCar dataset.
We use the same subsample of tracks and preprocessed point clouds as described in the
`PointNetVLAD paper <https://openaccess.thecvf.com/content_cvpr_2018/html/Uy_PointNetVLAD_Deep_Point_CVPR_2018_paper.html>`_.
Additionally, we created the files "train.csv", "val.csv", and "test.csv."

You can download our version of the dataset via the following link:

* [Kaggle](https://www.kaggle.com/datasets/creatorofuniverses/oxfordrobotcar-iprofi-hack-23)

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

.. include:: ../../LICENSE
