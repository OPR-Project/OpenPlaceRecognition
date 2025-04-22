import json
from os import PathLike
import heapq
import numpy as np
from typing import Dict, Tuple, List
from torch import Tensor, nn

import faiss
import torch
from scipy.spatial.transform import Rotation
from opr.pipelines.registration import PointcloudRegistrationPipeline, SequencePointcloudRegistrationPipeline
import MinkowskiEngine as ME


class TopologicalGraph():
    def __init__(self, 
                 place_recognition_model: nn.Module,
                 registration_pipeline: PointcloudRegistrationPipeline | SequencePointcloudRegistrationPipeline,
                 fusion: bool = False
                 ) -> None:
        self.vertices = []
        self.adj_lists = []
        self.model = place_recognition_model
        self.registration_pipeline = registration_pipeline
        if fusion:
            self.index = faiss.IndexFlatL2(512)
        else:
            self.index = faiss.IndexFlatL2(256)
        self._pointcloud_quantization_size = 0.5
        self.device = torch.device('cuda:0')

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
        return out_dict

    def normalize(self, angle: float) -> float:
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle > np.pi:
            angle -= 2 * np.pi
        return angle

    def load_from_json(self, input_path: str | PathLike) -> None:
        fin = open(input_path, 'r')
        j = json.load(fin)
        fin.close()
        self.vertices = j['vertices']
        self.adj_lists = j['edges']

    def add_vertex(self, x: float, y: float, theta: float, 
                   input_data: Dict[str, Tensor]) -> int:
        cloud = input_data['pointcloud_lidar_coords'].cpu().numpy()
        self.vertices.append((x, y, theta, cloud))
        self.adj_lists.append([])
        batch = self._preprocess_input(input_data)
        descriptor = self.model(batch)["final_descriptor"].detach().cpu().numpy()
        self.index.add(descriptor)
        # print('Add new vertex ({}, {}, {}) with idx {}'.format(x, y, theta, len(self.vertices) - 1))
        return len(self.vertices) - 1

    def get_k_most_similar(self, k: float,
                           cloud: np.ndarray, 
                           img_front: np.ndarray | None = None, 
                           img_back: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        input_data = {'pointcloud_lidar_coords': torch.Tensor(cloud[:, :3]).cuda(),
                    'pointcloud_lidar_feats': torch.ones((cloud.shape[0], 1)).cuda()}
        if img_front is not None:
            input_data['image_front'] = img_front
        if img_back is not None:
            input_data['image_back'] = img_back
        batch = self._preprocess_input(input_data)
        descriptor = self.model(batch)["final_descriptor"].detach().cpu().numpy()
        _, pred_i = self.index.search(descriptor, k)
        pred_i = pred_i[0]
        pred_tf = []
        for idx in pred_i:
            cand_x, cand_y, cand_theta, cand_cloud = self.vertices[idx]
            cand_cloud_tensor = torch.Tensor(cand_cloud[:, :3]).to(self.device)
            ref_cloud_tensor = torch.Tensor(cloud[:, :3]).to(self.device)
            tf_matrix = self.registration_pipeline.infer(ref_cloud_tensor, cand_cloud_tensor)
            tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
            tf_translation = tf_matrix[:3, 3]
            pred_tf.append(list(tf_rotation) + list(tf_translation))
        #     print('Tf rotation:', tf_rotation)
        #     print('Tf translation:', tf_translation)
        # print('Pred tf:', np.array(pred_tf))
        return pred_i, np.array(pred_tf)
    
    def inverse_transform(self, x: float, y: float, theta: float) -> List[float]:
        x_inv = -x * np.cos(theta) - y * np.sin(theta)
        y_inv = x * np.sin(theta) - y * np.cos(theta)
        theta_inv = -theta
        return [x_inv, y_inv, theta_inv]
    
    def add_edge(self, i: int, j: int, x: float, y: float, theta: float) -> None:
        # print('Add edge from ({}, {}) to ({}, {})'.format(self.vertices[i][0], self.vertices[i][1], self.vertices[j][0], self.vertices[j][1]))
        self.adj_lists[i].append((j, [x, y, theta]))
        self.adj_lists[j].append((i, self.inverse_transform(x, y, theta)))

    def get_vertex(self, vertex_id: int) -> Tuple[float, float, float, np.ndarray]:
        return self.vertices[vertex_id]

    def has_edge(self, u: int, v: int) -> bool:
        for x, _, __ in self.adj_lists[u]:
            if x == v:
                return True
        return False

    def get_path_with_length(self, u: int, v: int) -> Tuple[List | None, float]:
        # Initialize distances and previous nodes dictionaries
        distances = [float('inf')] * len(self.adj_lists)
        prev_nodes = [None] * len(self.adj_lists)
        # Set distance to start node as 0
        distances[u] = 0
        # Create priority queue with initial element (distance to start node, start node)
        heap = [(0, u)]
        # Run Dijkstra's algorithm
        while heap:
            # Pop node with lowest distance from heap
            current_distance, current_node = heapq.heappop(heap)
            if current_node == v:
                path = [current_node]
                cur = current_node
                while cur != u:
                    cur = prev_nodes[cur]
                    path.append(cur)
                path = path[::-1]
                return path, distances[v]
            # If current node has already been visited, skip it
            if current_distance > distances[current_node]:
                continue
            # For each neighbour of current node
            for neighbour, _, weight in self.adj_lists[current_node]:
                # Calculate tentative distance to neighbour through current node
                tentative_distance = current_distance + weight
                # Update distance and previous node if tentative distance is better than current distance
                if tentative_distance < distances[neighbour]:
                    distances[neighbour] = tentative_distance
                    prev_nodes[neighbour] = current_node
                    # Add neighbour to heap with updated distance
                    heapq.heappush(heap, (tentative_distance, neighbour))
        return None, float('inf')

    def save_to_json(self, output_path: str | PathLike) -> None:
        self.vertices = list(self.vertices)
        for i in range(len(self.vertices)):
            x, y, theta, cloud = self.vertices[i]
            self.vertices[i] = (x, y, theta)
        j = {'vertices': self.vertices, 'edges': self.adj_lists}
        fout = open(output_path, 'w')
        json.dump(j, fout)
        fout.close()