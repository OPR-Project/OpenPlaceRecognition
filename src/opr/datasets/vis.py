"""Visualisation utils for SOC dataset."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import plotly.graph_objects as go
from loguru import logger
from matplotlib.backends.backend_agg import FigureCanvasAgg

from opr.datasets.soc_utils import generate_color_sequence


class VisData:
    def __init__(self, staff_classes, output_dir=None):
        self.staff_classes = staff_classes
        self.staff_colors = generate_color_sequence(len(staff_classes), "Spectral")
        self.PLOTLY_COLORSCALE = [f"rgb({r},{g},{b})" for r, g, b in self.staff_colors]
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mask_count = 0
        self.projection_count = 0
        self.instance_count = 0

    @staticmethod
    def imshow(img: np.ndarray, mask=False):
        if not mask:
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def cloudshow(
        cloud,
        colors=None,
        save_path=None,
        colorscale=None,
        labels=None,
        staff_classes=None,
        staff_colors=None,
    ):
        colored = colors is not None
        if not colored:
            colors = cloud[:, 2]
            colorscale = "Viridis"
            label = None
        else:
            if colorscale is None:
                pass
                # colorscale = PLOTLY_COLORSCALE

        if colored:
            label = [staff_classes[int(idx)] for idx in colors]
            if labels is not None:
                label = [l_1 + "\n" + str(l_2) for l_1, l_2 in zip(label, labels)]  # type: ignore

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=cloud[:, 0],
                    y=cloud[:, 1],
                    z=cloud[:, 2],
                    mode="markers",
                    text=label,  # type: ignore
                    marker=dict(
                        size=2,
                        color=colors,  # set color to an array/list of desired values
                        colorscale=colorscale,  # choose a colorscale
                        opacity=0.8,
                    ),
                )
            ],
        )

        # tight layout
        fig.update_layout(scene_aspectmode="data")

        if save_path is None:
            fig.show()
        else:
            fig.write_image(save_path)

    def cloudshow_color(self, cloud, colors=None, save_path=None, colorscale=None, labels=None):
        if colorscale is None:
            colorscale = self.PLOTLY_COLORSCALE

        VisData.cloudshow(
            cloud,
            colors,
            save_path,
            colorscale,
            labels,
            self.staff_classes,
            self.staff_colors,
        )

    def get_colored_mask(self, img, mask, alpha=0.5, show=False, tag: str = "tag"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res_img = VisImage(img)

        labels, areas = np.unique(mask, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda label_id: label_id < len(self.staff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.staff_colors[int(label)]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (mask == label).astype(np.uint8)
            res_img = self._draw_binary_mask(
                binary_mask,
                color=mask_color,
                res_img=res_img,
                alpha=alpha,
            )

        if show:
            cv2.imshow("image", res_img.get_image())
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # save image
            output_dir = self.output_dir / "colored_masks"
            output_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_dir / f"mask_{tag}_{self.mask_count}.png"), res_img.get_image())
            self.mask_count += 1

        return res_img

    def _draw_binary_mask(self, binary_mask, color=None, *, res_img, alpha=0.5):
        """Draw a binary mask in a given color channel.

        Args:
            binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
                W is the image width. Each value in the array is either a 0 or 1 value of uint8
                type.
            color: color of the mask. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If None, will pick a random color.
            edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
                full list of formats that are accepted.
            text (str): if None, will be drawn on the object
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            area_threshold (float): a connected component smaller than this area will not be shown.

        Returns:
            output (VisImage): image object with mask drawn.
        """
        color = mplc.to_rgb(color)  # type: ignore

        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        # mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
        res_img.ax.imshow(rgba, extent=(0, res_img.width, res_img.height, 0))

        return res_img

    def draw_points_on_image(self, img: np.ndarray, points: np.ndarray, labels: np.ndarray, tag: str = "tag"):
        """Args:
            img (ndarray): standart opencv image.

            points (ndarray): array of 2D coordinates of projected points with shape (n, 2).
            Coordinates should match with cam_resolution.

            colors (ndarray): array of colors for each point in RGB uint8 format with shape (n, 3).

        Returns:
            img (ndarray): standart opencv image with points drawn.
        """
        proj_img = img.copy()

        for point, label in zip(points.T, labels):  # points.T
            c = [
                int(round(255 * x)) for x in self.staff_colors[int(label)]
            ]  # (int(color[0]), int(color[1]), int(color[2]))
            proj_img = cv2.circle(proj_img, point, radius=2, color=c, thickness=cv2.FILLED)

        output_dir = self.output_dir / "projections"
        output_dir.mkdir(parents=True, exist_ok=True)
        proj_img = cv2.cvtColor(proj_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(output_dir / f"projection_{tag}_{self.projection_count}.png"), proj_img)
        self.projection_count += 1

        return proj_img

    @staticmethod
    def get_points_labels_by_mask(points: np.ndarray, mask: np.ndarray):
        """Args:
            points (ndarray): array of 2D coordinates of projected points with shape (n, 2).
            Coordinates should match with cam_resolution.

            mask (ndarray): semantic mask in opencv  image format (ndarray)

        Returns:
            labels (ndarray): point labels taken from the mask.
        """
        labels = []
        for img_point in points.T:  # points.T
            labels.append(mask[img_point[1], img_point[0]])  # ! Magic number
        # ? Because of Unknown and Dynamic-by-Motion labels added

        return np.asarray(labels)

    def draw_instances(self, img, mask, classes, area_threshold, bboxes=True, tag: str = "tag"):
        """Draw instance labels from semantic mask.
        Instances are defined as connected components of the same class.
        Connected components found using opencv connectedComponentsWithStats opencv algorithm
        in class-wise manner.

        Args:
            mask (ndarray): semantic mask in opencv  image format (ndarray)

        Returns:
            None

        """
        class_labels = [self.staff_classes.index(c) for c in classes]

        instances = {}
        for label in class_labels:
            instances[label] = []
            binary_mask = (mask == label).astype(np.uint8)
            (
                totalLabels,
                label_ids,
                stats,
                centroid,
            ) = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            logger.debug(f"Label: {self.staff_classes[int(label)]}")
            logger.debug(f"Total labels: {totalLabels}")
            logger.debug(f"Label ids: {label_ids}")
            logger.debug(f"Label ids uniq: {np.unique(label_ids)}")
            logger.debug(f"Stats: {stats}")

            components = []
            for label_id in range(1, totalLabels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area > area_threshold:
                    components.append(label_ids == label_id)
                    logger.debug(f"Area: {area}")

            instances[label] = components

        # logger.debug(f"Instances: {instances}")
        result = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        for label, masks in instances.items():
            instance_colors = generate_color_sequence(len(masks), "rocket")
            for idx, mask in enumerate(masks):
                res_img = VisImage(result)
                res_img = self._draw_binary_mask(
                    binary_mask,
                    color=[x / 255 for x in self.staff_colors[int(label)]],
                    res_img=res_img,
                    alpha=0.25,
                )
                # to make instances look different
                res_img = self._draw_binary_mask(
                    binary_mask,
                    color=instance_colors[idx],
                    res_img=res_img,
                    alpha=0.25,
                )
                result = res_img.get_image()
                if bboxes:
                    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                    cv2.rectangle(
                        result,
                        (x, y),
                        (x + w, y + h),
                        [int(round(255 * x)) for x in self.staff_colors[int(label)]],
                        2,
                    )
                    # draw class name
                    cv2.rectangle(
                        result,
                        (x, y - 20),
                        (x + w, y),
                        [int(round(255 * x)) for x in self.staff_colors[int(label)]],
                        -1,
                    )
                    text_label = f"{self.staff_classes[int(label)]}"
                    cv2.putText(
                        result,
                        text_label,
                        (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        output_dir = self.output_dir / "instances"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"instance_{tag}_{self.instance_count}.png"), result)
        self.instance_count += 1


class VisImage:
    def __init__(self, img, scale=1.0):
        """Args:
        img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
        scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """Args:
        img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """Args:
        filepath (str): a string that contains the absolute path, including the file name, where
            the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """Returns:
        ndarray:
            the visualized image of shape (H, W, 3) (RGB) in uint8 type.
            The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")
