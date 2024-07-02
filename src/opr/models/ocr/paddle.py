import logging
import paddleocr.tools.infer.predict_rec as predict_rec
import paddleocr.tools.infer.predict_det as predict_det
import paddleocr.tools.infer.predict_cls as predict_cls
from paddleocr.tools.infer.predict_system import TextSystem
from paddleocr.ppocr.utils.logging import get_logger


logger = get_logger()


class PaddleOcrPipeline(TextSystem):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0