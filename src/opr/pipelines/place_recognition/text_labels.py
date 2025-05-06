import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from fuzzywuzzy import fuzz
from torch import Tensor, nn

from opr.pipelines.place_recognition.base import PlaceRecognitionPipeline
from opr.models.ocr.paddle import PaddleOcrPipeline

class TextLabelsPlaceRecognitionPipeline(PlaceRecognitionPipeline):
    def __init__(self, db_labels_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(db_labels_path, "rb") as f:
            db_labels = json.load(f)
            if isinstance(db_labels, str):
                db_labels = json.loads(db_labels)

        self.db_labels = db_labels

    @staticmethod
    def get_labels_by_id(labels: List[str], id: str) -> List[str]:
        """
        Retrieve all labels associated with a given ID from the labels dictionary.

        Args:
            labels (List[str]): The list of labels.
            id (str): The ID to retrieve labels for.

        Returns:
            List[str]: The list of labels associated with the given ID.
        """
        frame = labels[id]
        all_labels = [i["value"]["text"] for i in frame["back_cam_anno"] + frame["front_cam_anno"]]
        all_labels = sum(all_labels, [])
        return all_labels

    @staticmethod
    def normalize_labels(labels: List[str]) -> List[str]:
        """
        Normalize a list of labels by converting them to lowercase and removing leading/trailing whitespace.

        Args:
            labels (List[str]): The list of labels to be normalized.

        Returns:
            List[str]: The normalized list of labels.
        """
        norm_labels = [i.lower() for i in labels]
        norm_labels = [i.strip() for i in norm_labels]
        return norm_labels

    @staticmethod
    def remove_stopwords(labels: List[str], stopwords: List[str] = ["выход", "мфти"]) -> List[str]:
        """
        Removes stopwords from a list of labels.

        Args:
            labels (List[str]): The list of labels to remove stopwords from.
            stopwords (List[str], optional): The list of stopwords to be removed. Defaults to ["выход", "мфти"].

        Returns:
            List[str]: The list of labels with stopwords removed.
        """
        return [i for i in labels if i not in stopwords]

    def find_most_similar_id(
        self,
        query: List[str],
        ignore_stopwords: bool = False,
        normalize_text: bool = False,
        print_info: bool = False,
    ) -> Tuple[str, List[str], int]:
        """
        Finds the most similar ID in the database based on the given query.

        Args:
            query (List[str]): The query to compare against the database labels.
            ignore_stopwords (bool, optional): Whether to ignore stopwords during comparison. Defaults to False.
            normalize_text (bool, optional): Whether to normalize the text before comparison. Defaults to False.
            print_info (bool, optional): Whether to print additional information during the process. Defaults to False.

        Returns:
            Tuple[Optional[str], Optional[List[str]], int]: A tuple containing the best match ID, the corresponding labels,
            and the highest similarity score.
        """

        if normalize_text:
            query = self.normalize_labels(query)
        if ignore_stopwords:
            query = self.remove_stopwords(query)

        if print_info:
            print(f"query: {query}")

        best_match_id = None
        best_match_annos = None
        highest_similarity = 0

        for db_key in self.db_labels.keys():
            db_frame = self.get_labels_by_id(self.db_labels, db_key)

            if normalize_text:
                db_frame = self.normalize_labels(db_frame)
            if ignore_stopwords:
                db_frame = self.remove_stopwords(db_frame)

            # Calculate the similarity between the database item and the query
            similarity = fuzz.token_set_ratio(query, db_frame)

            # Update the best match if the current comparison is better than the previous one
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = db_key
                best_match_annos = db_frame

        if print_info:
            print(f"best_match_annos: {best_match_annos}, highest_similarity: {highest_similarity}")

        return best_match_id, best_match_annos, highest_similarity

    def infer(
        self, input_data: Dict[str, Tensor], query_labels: List[str], text_similarity_thresh: int = 50, print_info: bool = False
    ) -> Dict[str, np.ndarray]:
        """Single sample inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.

                "query_labels"  List of query labels.

                "text_similarity_thresh"  Text similarity threshold. Defaults to 50.

        Returns:
            Dict[str, np.ndarray]: Inference results. Dictionary with keys:

                "idx" for predicted index in the database,

                "pose" for predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "descriptor" for predicted descriptor.
        """

        best_match_id, best_match_annos, highest_similarity = self.find_most_similar_id(
            query_labels, normalize_text=True, ignore_stopwords=True
        )

        input_data = self._preprocess_input(input_data)
        output = {}
        with torch.no_grad():
            descriptor = self.model(input_data)["final_descriptor"].cpu().numpy()

        if highest_similarity > text_similarity_thresh:
            search_df = self.database_df.reset_index() # в исходном датафрейме скипнуты индексы
            # pred_i = self.database_df[self.database_df["timestamp"] == int(best_match_id)].index[0]
            pred_i = search_df[search_df["timestamp"] == int(best_match_id)].index[0]
            if print_info:
                print("Using text labels")
                print(f"pred_i: {pred_i}, best_match_id: {best_match_id}")
                print(f"best_match_annos: {best_match_annos}, highest_similarity: {highest_similarity}")
        else:
            _, pred_i = self.database_index.search(descriptor, 1)
            pred_i = pred_i[0][0]
            if print_info:
                print(f"pred_i: {pred_i}")
                print("Using image descriptors")

        pred_pose = self.database_df.iloc[pred_i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(
            dtype=float
        )
        output["idx"] = pred_i
        output["pose"] = pred_pose
        output["descriptor"] = descriptor[0]
        return output


class TextLabelsPlaceRecognitionOCRPipeline(PlaceRecognitionPipeline):
    def __init__(self, db_labels_path, *args, **kwargs):

        super().__init__(*args, **kwargs)

        with open(db_labels_path, "rb") as f:
            db_labels = json.load(f)
            # db_labels = json.loads(db_labels)

        self.db_labels = db_labels
        self.ocr_model = None

    def init_ocr_model(self, ocr_model):
        self.ocr_model = ocr_model

    @staticmethod
    def get_labels_by_id(labels: List[str], id: str) -> List[str]:
        """
        Retrieve all labels associated with a given ID from the labels dictionary.

        Args:
            labels (List[str]): The list of labels.
            id (str): The ID to retrieve labels for.

        Returns:
            List[str]: The list of labels associated with the given ID.
        """
        frame = labels[id]
        all_labels = [i["value"]["text"] for i in frame["back_cam_anno"] + frame["front_cam_anno"]]
        all_labels = sum(all_labels, [])
        return all_labels

    @staticmethod
    def normalize_labels(labels: List[str]) -> List[str]:
        """
        Normalize a list of labels by converting them to lowercase and removing leading/trailing whitespace.

        Args:
            labels (List[str]): The list of labels to be normalized.

        Returns:
            List[str]: The normalized list of labels.
        """
        norm_labels = [i.lower() for i in labels]
        norm_labels = [i.strip() for i in norm_labels]
        return norm_labels

    @staticmethod
    def remove_stopwords(labels: List[str], stopwords: List[str] = ["выход", "мфти"]) -> List[str]:
        """
        Removes stopwords from a list of labels.

        Args:
            labels (List[str]): The list of labels to remove stopwords from.
            stopwords (List[str], optional): The list of stopwords to be removed. Defaults to ["выход", "мфти"].

        Returns:
            List[str]: The list of labels with stopwords removed.
        """
        return [i for i in labels if i not in stopwords]

    def find_most_similar_id(
        self,
        query: List[str],
        ignore_stopwords: bool = False,
        normalize_text: bool = False,
        print_info: bool = False,
    ) -> Tuple[str, List[str], int]:
        """
        Finds the most similar ID in the database based on the given query.

        Args:
            query (List[str]): The query to compare against the database labels.
            ignore_stopwords (bool, optional): Whether to ignore stopwords during comparison. Defaults to False.
            normalize_text (bool, optional): Whether to normalize the text before comparison. Defaults to False.
            print_info (bool, optional): Whether to print additional information during the process. Defaults to False.

        Returns:
            Tuple[Optional[str], Optional[List[str]], int]: A tuple containing the best match ID, the corresponding labels,
            and the highest similarity score.
        """

        if normalize_text:
            query = self.normalize_labels(query)
        if ignore_stopwords:
            query = self.remove_stopwords(query)

        if print_info:
            print(f"query: {query}")

        best_match_id = None
        best_match_annos = None
        highest_similarity = 0

        for db_key in self.db_labels.keys():
            db_frame = self.get_labels_by_id(self.db_labels, db_key)

            if normalize_text:
                db_frame = self.normalize_labels(db_frame)
            if ignore_stopwords:
                db_frame = self.remove_stopwords(db_frame)

            # Calculate the similarity between the database item and the query
            similarity = fuzz.token_set_ratio(query, db_frame)

            # Update the best match if the current comparison is better than the previous one
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = db_key
                best_match_annos = db_frame

        if print_info:
            print(f"best_match_annos: {best_match_annos}, highest_similarity: {highest_similarity}")

        return best_match_id, best_match_annos, highest_similarity


    def infer(
        self, input_data: Dict[str, Tensor], text_similarity_thresh: int = 50, print_info: bool = False
    ) -> Dict[str, np.ndarray]:
        """Single sample inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.

                "text_similarity_thresh"  Text similarity threshold. Defaults to 50.

        Returns:
            Dict[str, np.ndarray]: Inference results. Dictionary with keys:

                "idx" for predicted index in the database,

                "pose" for predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "descriptor" for predicted descriptor.
        """

        query_labels = []
        for key in input_data:
            if "image_" in key:
                # opened_image = cv2.imread(input_data[key])
                tensor = input_data[key]
                tensor = (tensor + 1) * 127.5
                image = tensor.clamp(0, 255).byte().cpu().detach().numpy().transpose(1, 2, 0)
                filtered_boxes, texts, time_stats = self.ocr_model(image)
                texts = [text for text, prob in texts]
                query_labels.extend(texts)
        best_match_id, best_match_annos, highest_similarity = self.find_most_similar_id(
            query_labels, normalize_text=True, ignore_stopwords=True
        )

        input_data = self._preprocess_input(input_data)
        output = {}
        with torch.no_grad():
            descriptor = self.model(input_data)["final_descriptor"].cpu().numpy()

        if highest_similarity > text_similarity_thresh:
            search_df = self.database_df.reset_index() # в исходном датафрейме скипнуты индексы
            # pred_i = self.database_df[self.database_df["timestamp"] == int(best_match_id)].index[0]
            pred_i = search_df[search_df["timestamp"] == int(best_match_id)].index[0]
            if print_info:
                print("Using text labels")
                print(f"pred_i: {pred_i}, best_match_id: {best_match_id}")
                print(f"best_match_annos: {best_match_annos}, highest_similarity: {highest_similarity}")
        else:
            _, pred_i = self.database_index.search(descriptor, 1)
            pred_i = pred_i[0][0]
            if print_info:
                print(f"pred_i: {pred_i}")
                print("Using image descriptors")

        pred_pose = self.database_df.iloc[pred_i][["tx", "ty", "tz", "qx", "qy", "qz", "qw"]].to_numpy(
            dtype=float
        )
        output["idx"] = pred_i
        output["pose"] = pred_pose
        output["descriptor"] = descriptor[0]
        return output
