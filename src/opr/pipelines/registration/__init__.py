"""Registration pipelines."""
from .pointcloud import (
    PointcloudRegistrationPipeline,
    RansacGlobalRegistrationPipeline,
    SequencePointcloudRegistrationPipeline,
    #Feature2DGlobalRegistrationPipeline
)
from .occupancy_grid import (
    Feature2DGlobalRegistrationPipeline
)
