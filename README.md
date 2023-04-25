# Open Place Recognition library

## Installation

### Pre-requisites

- The library requires PyTorch~=1.13 and MinkowskiEngine library to be installed manually. See [PyTorch website](https://pytorch.org/get-started/previous-versions/) and [MinkowskiEngine repository](https://github.com/NVIDIA/MinkowskiEngine) for the detailed instructions.

- Another option is to use the suggested Dockerfile. The following commands should be used to build, start and enter the container:

  1. Build the image

      ```bash
      bash docker/build.sh
      ```

  2. Start the container with the datasets directory mounted:

      ```bash
      bash docker/start.sh [DATASETS_DIR]
      ```

  3. Enter the container (if needed):

      ```bash
      bash docker/into.sh
      ```

### Library installation

- After the pre-requisites are met, install the Open Place Recognition library with the following command:

    ```bash
    pip install .
    ```

## Usage

Currently only MinkLoc++ pretrained on Oxford RobotCar available. You can download it using [google drive link](https://drive.google.com/file/d/1zlfdX217Nh3_QL5r0XAHUjDFjIPxUmMg/view?usp=share_link) (the link is subject to change).

If everything is installed correctly, you can use the library like below:

```python
from opr.models import minkloc_multimodal

baseline_model = minkloc_multimodal(weights="path_to_checkpoint")
```

## License

[MIT License](./LICENSE) (**_the license is subject to change in future versions_**)
