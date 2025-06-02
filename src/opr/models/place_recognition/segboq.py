import torch
from torch import nn
from opr.modules.feature_extractors.dinov2 import ViTBaseFeatureExtractor
from opr.modules.feature_extractors.dinov2 import DINO_V2_MODELS, DINO_FACETS, BOQ_MODELS
from typing import Union, Dict, List
from torch import Tensor
from scipy.spatial import Delaunay
from opr.datasets.augmentations import DefaultImageTransform
from opr.utils import parse_device


class SegBoQ(nn.Module):

    feature_extractor: ViTBaseFeatureExtractor
    device: torch.device
    segments_radius: int
    segments_image_transform: DefaultImageTransform
    segments_limit: int
    use_whole_image: bool

    def __init__(
            self,
            feature_extractor_model: str, #Union[DINO_V2_MODELS, BOQ_MODELS],
            feature_extractor_layer: int,
            feature_extractor_facet: DINO_FACETS="token",
            feature_extractor_use_cls=False,
            feature_extractor_norm_descs=True,
            device: str = "cpu",
            segments_radius: int = 1,
            segments_limit: int = 10,
            segments_image_transform = DefaultImageTransform(resize=(322, 322), train=False),
            use_whole_image: bool = False
        ) -> None:
        """
        Initialize the SegBoQ model.

        Args:
            feature_extractor_model (str): Name of the vision transformer model to use for feature extractor.
            feature_extractor_layer (int): Layer index from which to extract features for feature extractor.
            feature_extractor_facet (DINO_FACETS, optional): Feature facet to extract (e.g., 'token' or 'block') for feature extractor. Defaults to "token".
            feature_extractor_use_cls (bool, optional): Whether to use the [CLS] token as a feature for feature extractor. Defaults to False.
            feature_extractor_norm_descs (bool, optional): Whether to normalize feature descriptors for feature extractor. Defaults to True.
            device (str, optional): Device to run computations on (e.g., "cpu" or "cuda"). Defaults to "cpu".
            segments_radius (int, optional): Radius used to define the image segments. Defaults to 2.
            segments_limit (int, optional): Maximum number of segments to extract. Defaults to 10.
            segments_image_transform (callable, optional): Transformation applied to image segments before processing. Defaults to `DefaultImageTransform(resize=(322, 322), train=False)`.
            use_whole_image (bool, optional): Whether to include the whole image as an additional segment. Defaults to False.

        Returns:
            None
        """

        super().__init__()
        self.feature_extractor = ViTBaseFeatureExtractor(
            vit_type=feature_extractor_model,
            layer=feature_extractor_layer,
            facet=feature_extractor_facet,
            use_cls=feature_extractor_use_cls,
            norm_descs=feature_extractor_norm_descs,
            device=device,
        )
        self.device = parse_device(device)
        self.segments_radius = segments_radius
        self.segments_image_transform = segments_image_transform
        self.segments_limit = segments_limit
        self.use_whole_image = use_whole_image


    def forward(self, batch: Dict[str, Union[Tensor, List[Tensor]]]) -> Dict[str, List[Tensor]]: # TODO make normal batch processing
        """
        Forward pass of the SegBoQ model.

        Args:
            batch (Dict[str, Union[Tensor, List[Tensor]]]): Input batch.

        Returns:
            Dict[str, List[Tensor]]: Dictionary containing the final descriptor for each segment.
        """

        segments_features: List[List[Tensor]] = []
        for key in batch.keys():
            if not key.startswith("bounding_boxes_"):
                continue
            for ind_in_batch, bounding_boxes in enumerate(batch[key]):
                objects_graph = self._create_objects_graph(bounding_boxes)

                selected_vertices = self._greedy_sparse_subgraph(objects_graph, self.segments_limit)

                # Compute the power of the graph to find connected components
                n = bounding_boxes.shape[0]
                graph_power = torch.eye(n, n, device=self.device, dtype=torch.float32)
                graph = torch.eye(n, n, device=self.device, dtype=torch.float32)
                for i in range(self.segments_radius):
                    graph_power = torch.matmul(graph_power, objects_graph)
                    graph += graph_power
                graph = (graph > 0).int()
                
                segments_bounding_boxes = self._calculate_segment_bounding_boxes(graph, bounding_boxes, selected_vertices)
                segment_images = []
                image = batch[f"images_{key[15:]}"]
                if self.use_whole_image:
                    segments_bounding_boxes.append(Tensor([0, 0, image.shape[2], image.shape[1]]))
                for segment_bounding_box in segments_bounding_boxes:
                    # Extract the bounding box region from the image
                    x_min, y_min, x_max, y_max = segment_bounding_box
                    segment_image = image[ind_in_batch, int(y_min):int(y_max), int(x_min):int(x_max), :]
                    # Apply the image transform
                    segment_image = self.segments_image_transform(segment_image.cpu().numpy()).to(self.device)
                    segment_images.append(segment_image)

                # Extract features using the feature extractor
                segment_features = self.feature_extractor(torch.stack(segment_images))
                segments_features.append(segment_features)


        out_dict: Dict[str, List[List[Tensor]]] = {"final_descriptor": segments_features}
        return out_dict


    @staticmethod
    def _compute_centroids(bounding_boxes: Tensor) -> Tensor:
        """
        Compute the centroids of bounding boxes.

        Args:
            bounding_boxes (Tensor): Tensor of shape (N, 4) where each row represents
                                            [x_min, y_min, x_max, y_max].

        Returns:
            Tensor: Tensor of shape (N, 2) where each row represents the centroid [x, y].
        """

        # Split the tensor into min and max coordinates
        min_coords = bounding_boxes[:, :2]  # (N, 2) -> [x_min, y_min]
        max_coords = bounding_boxes[:, 2:]  # (N, 2) -> [x_max, y_max]

        # Compute centroids as the midpoint between min and max coordinates
        centroids = (min_coords + max_coords) / 2

        return centroids
    

    def _create_objects_graph(self, bounding_boxes: Tensor) -> Tensor:
        """
        Create a graph representation of the objects in the scene.

        Args:
            bounding_boxes (Tensor): Tensor of shape (N, 4) where each row represents
                                            [x_min, y_min, x_max, y_max].

        Returns:
            Tensor: Adjacency matrix representing the graph.
        """

        n = bounding_boxes.shape[0]
        centroids = self._compute_centroids(bounding_boxes)

        try:
            tri = Delaunay(centroids.cpu().numpy())
        except Exception as e:
            return torch.eye(n, n, device=self.device, dtype=torch.float32)

        # Create a graph adjacency matrix
        graph = torch.eye(n, n, device=self.device, dtype=torch.float32)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    graph[simplex[i], simplex[j]] = 1
                    graph[simplex[j], simplex[i]] = 1
        
        return graph
    

    def _calculate_segment_bounding_boxes(self, graph: Tensor, bounding_boxes: Tensor, selected_vertices: List[int]) -> List[Tensor]:
        """
        Calculate the bounding boxes for each segment in the graph.

        Args:
            graph (Tensor): Adjacency matrix representing the graph.
            bounding_boxes (Tensor): Tensor of shape (N, 4) where each row represents
                                            [x_min, y_min, x_max, y_max].

        Returns:
            List[Tensor]: List of tensors representing the bounding boxes for each segment.
        """
        segments_bounding_boxes = []

        for i in selected_vertices:
            # Find all nodes connected to node i
            connected_nodes = torch.nonzero(graph[i], as_tuple=False).squeeze(1)
            # Get the bounding boxes for the connected nodes
            segment_boxes = bounding_boxes[connected_nodes]
            # Calculate the combined bounding box for the segment
            min_coords = torch.min(segment_boxes[:, :2], dim=0).values
            max_coords = torch.max(segment_boxes[:, 2:], dim=0).values
            segments_bounding_boxes.append(torch.cat((min_coords, max_coords)))

        return segments_bounding_boxes


    @staticmethod
    def _greedy_sparse_subgraph(adj_matrix: Tensor, k: int):
        """
        Selects k vertices from a graph (given as an adjacency matrix)
        such that the induced subgraph is as sparse as possible.
        
        Parameters:
            adj_matrix (np.ndarray): n x n adjacency matrix (0-1, symmetric).
            k (int): number of vertices to select.
        
        Returns:
            list: indices of selected vertices.
        """
        n = adj_matrix.shape[0]

        if n <= k:
            return list(range(n))

        selected = []
        remaining = torch.ones(n, dtype=torch.bool, device=adj_matrix.device)
        connection_scores = torch.zeros(n, device=adj_matrix.device)

        for _ in range(k):
            if selected:
                # Update connection scores: sum of edges to selected nodes
                connection_scores = adj_matrix[:, selected].sum(dim=1)

            # Mask out already selected nodes
            connection_scores[~remaining] = float('inf')

            # Pick node with fewest connections to selected
            best_vertex = torch.argmin(connection_scores).item()
            selected.append(best_vertex)
            remaining[best_vertex] = False

        return selected
