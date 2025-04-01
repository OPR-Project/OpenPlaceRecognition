ITLPCampus Dataset Reader
=========================

A module that implements a PyTorch dataset reader for the ITLPCampus dataset.

More details about the dataset can be found on the
:doc:`../itlp_dataset` page.

Below is a usage example of the dataset reader that is implemented
as part of the OpenPlaceRecognition library.


Usage example
-------------

You should start with initializing the dataset reader by creating an instance of the
:class:`opr.datasets.itlp.ITLPCampus` class with appropriate parameters.

.. code-block:: python

   from opr.datasets.itlp import ITLPCampus
   from pathlib import Path
   from torch.utils.data import DataLoader

   # Path to your ITLP dataset directory
   track_dir = Path("/path/to/ITLP_Campus_outdoor/00_2023-02-21")  # outdoor track example

   # Initialize the dataset reader
   dataset = ITLPCampus(
       dataset_root=track_dir,                      # track directory
       sensors=["front_cam", "back_cam", "lidar"],  # sensors to load
       load_semantics=True,                         # load semantic masks
       mink_quantization_size=0.5,                  # point cloud voxelization size
       max_point_distance=None,                     # max distance to keep points (None for no limit)
       indoor=False,                                # specify if it's an indoor track
   )

   # Access a single sample from the dataset
   sample = dataset[0]  # returns a dictionary with data from the first frame

The ``sample`` will contain a dictionary with the following keys:

* ``"idx"``: index of the sample in the dataset
* ``"pose"``: 7-element pose vector [tx, ty, tz, qx, qy, qz, qw]
* ``"image_front_cam"``: tensor of front camera image (if "front_cam" in sensors)
* ``"image_back_cam"``: tensor of back camera image (if "back_cam" in sensors)
* ``"mask_front_cam"``: tensor of front camera semantic mask (if load_semantics=True)
* ``"mask_back_cam"``: tensor of back camera semantic mask (if load_semantics=True)
* ``"pointcloud_lidar_coords"``: tensor of LiDAR point coordinates (if "lidar" in sensors)
* ``"pointcloud_lidar_feats"``: tensor of LiDAR point features (if "lidar" in sensors)

To create a DataLoader for batch training or inference:

.. code-block:: python

   # Create a DataLoader using the dataset's collate_fn
   batch_size = 4
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=True,
       collate_fn=dataset.collate_fn,
       num_workers=4
   )

   # Iterate through the data batches
   for batch in dataloader:
       # Process batch data
       # batch["images_front_cam"] # [batch_size, 3, H, W]
       # batch["images_back_cam"]  # [batch_size, 3, H, W]
       # batch["masks_front_cam"]  # [batch_size, 1, H, W]
       # batch["poses"]            # [batch_size, 7]
       # batch["pointclouds_lidar_coords"] # ME batched coordinates format
       # batch["pointclouds_lidar_feats"]  # ME batched features format
       pass

For indoor datasets, you can specify train/test splits:

.. code-block:: python

   # Split floors for indoor dataset
   train_split = ["1", "2", "3"]  # floor numbers for training
   test_split = ["4", "5"]        # floor numbers for testing

   # Create training dataset
   train_dataset = ITLPCampus(
       dataset_root=track_dir,
       subset="train",              # "train", "val", or "test"
       indoor=True,
       train_split=train_split,
       test_split=test_split
   )

   # Create test dataset
   test_dataset = ITLPCampus(
       dataset_root=track_dir,
       subset="test",
       indoor=True,
       train_split=train_split,
       test_split=test_split
   )

More examples of using the ``ITLPCampus`` dataset reader can be found in the notebooks:

* `notebooks/demo_itlp_dataset.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/demo_itlp_dataset.ipynb>`_
