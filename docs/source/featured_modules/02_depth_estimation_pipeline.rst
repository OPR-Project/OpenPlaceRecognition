DepthEstimationPipeline
========================

A method that implements depth map reconstruction from a monocular image by a neural network and scaling of the reconstructed depth map using a sparse lidar point cloud.

Usage example
-------------

This is an example of indoor depth reconstruction with state-of-the-art (2024) DepthAnything-v2 neural network model.
At start, you should first initialize depth estimation neural network model, which can be imported from `the DepthAnything-v2 package which is added to OPR as a submodule <https://github.com/DepthAnything/Depth-Anything-V2/tree/28ad5a0797dfb8ac76d1e3dcddbe2160cbcc6c8d>`_:

.. code-block:: python

  import sys
  sys.path.append('<path_to_opr>/third_party/Depth-Anything-v2')
  from metric_depth.depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Metric

For the best performance, we recommend to use version "small" of the model:

.. code-block:: python

  import torch
  import os
  MODELS_BASE_PATH = <path to the DepthAnything-v2 model weights>
  model_configs = {
      'small': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
      'base': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
      'large': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
  }
  type = 'small'
  params = model_configs[type]
  new_model = DepthAnythingV2Metric(**params, max_depth=20.0)
  model_path=os.path.join(MODELS_BASE_PATH, 'depth_anything_v2_metric_hypersim_vits.pth')
  new_model.load_state_dict(torch.load(model_path))
  new_model.to(DEVICE)

Next, you should create an instance of DepthEstimationPipeline. It requires a camera intrinsic matrix and a lidar-to-camera transformation matrix for correct operation. This is an example of the matrices for ITLP-Campus dataset:

.. code-block:: python

  fx = 683.6
  fy = fx
  cx = 615.1
  cy = 345.3
  camera_matrix = {'f': fx, 'cx': cx, 'cy': cy}
  rotation = [-0.498, 0.498, -0.495, 0.510]
  R = Rotation.from_quat(rotation).as_matrix()
  #R = np.linalg.inv(R)
  translation = np.array([[0.061], [0.049], [-0.131]])
  tf_matrix = np.concatenate([R, translation], axis=1)
  tf_matrix = np.concatenate([tf_matrix, np.array([[0, 0, 0, 1]])], axis=0)

Create the DepthEstimationPipeline with these matrices:

.. code-block:: python

  pipeline = DepthEstimationPipeline(new_model, model_type='DepthAnything', align_type='average', mode='outdoor')
  pipeline.set_camera_matrix(camera_matrix)
  pipeline.set_lidar_to_camera_transform(tf_matrix)

Now you can run the pipeline on a pair of an RGB image and a lidar point cloud:

.. code-block:: python

  predicted_depth = de_new.get_depth_with_lidar(test_img, test_cloud)

* ``"test_img"`` is an RGB image stored as numpy.ndarray of shape (h, w, 3) and uint8 data type
* ``"test_cloud"`` is a lidar point cloud stored as numpy.ndarray of shape (N, 3) and float data type (the values are (x, y, z) coordinates in meters)
* ``"predicted_depth"`` is the reconstructed depth stored as numpy.ndarray of shape (h, w) and float data type. The values of the array are depths in meters.

An illustrated demo of the work of DepthEstimationPipeline can be found in 
`the demo notebook <https://github.com/OPR-Project/OpenPlaceRecognition/blob/depth_reconstruction_nclt/notebooks/test_depth_reconstruction.ipynb>`_
