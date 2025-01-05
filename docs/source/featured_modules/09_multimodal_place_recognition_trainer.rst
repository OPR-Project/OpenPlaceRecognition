MultimodalPlaceRecognitionTrainer
=================================

A module that implements a training algorithm for a multimodal neural network model of global localization
based on the contrastive learning approach.


Usage example
-------------

You should initialize the
:class:`opr.trainers.place_recognition.multimodal.MultimodalPlaceRecognitionTrainer`
class with the desired parameters:

.. code:: python

   from opr.trainers.place_recognition.multimodal import MultimodalPlaceRecognitionTrainer

   trainer = MultimodalPlaceRecognitionTrainer(
       modalities_weights=modalities_weights,  # dictionary like {"image": 1.0, "cloud": 1.0}
       checkpoints_dir=checkpoints_dir,
       model=model,
       loss_fn=loss_fn,
       optimizer=optimizer,
       scheduler=scheduler,
       batch_expansion_threshold=batch_expansion_threshold,  # value in range [0, 1]
       wandb_log=True,  # or False
       device="cuda",
   )

To start training, you should call the
:meth:`opr.trainers.place_recognition.multimodal.MultimodalPlaceRecognitionTrainer.train`
method:

.. code:: python

   trainer.train(
       epochs=100,
       train_dataloader=train_dataloader,
       val_dataloader=train_dataloader,
       test_dataloader=train_dataloader,
   )

If you want to test the model on a test dataset, you should call the
:meth:`opr.trainers.place_recognition.multimodal.MultimodalPlaceRecognitionTrainer.test`
method:

.. code:: python

   trainer.test(test_dataloader)

More usage examples can be found in the following scripts and notebooks:

* `scripts/training/place_recognition/train_multimodal.py <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/scripts/training/place_recognition/train_multimodal.py>`_
* `scripts/training/place_recognition/find_parameters_multimodal.py <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/scripts/training/place_recognition/find_parameters_multimodal.py>`_
* `notebooks/finetune_itlp/finetune_itlp_multimodal.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/finetune_itlp/finetune_itlp_multimodal.ipynb>`_
* `notebooks/finetune_itlp/finetune_itlp_multimodal_semantic.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/finetune_itlp/finetune_itlp_multimodal_semantic.ipynb>`_
* `notebooks/finetune_itlp/finetune_itlp_multimodal_semantic_with_soc_outdoor.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/finetune_itlp/finetune_itlp_multimodal_semantic_with_soc_outdoor.ipynb>`_
* `notebooks/finetune_itlp/finetune_itlp_multimodal_with_soc_outdoor.ipynb <https://github.com/OPR-Project/OpenPlaceRecognition/blob/main/notebooks/finetune_itlp/finetune_itlp_multimodal_with_soc_outdoor.ipynb>`_
