import logging

from opr.optional_deps import lazy

paddleocr = lazy("paddleocr", feature="OCR")

logger = paddleocr.ppocr.utils.logging.get_logger()


class PaddleOcrPipeline(paddleocr.tools.infer.predict_system.TextSystem):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = paddleocr.tools.infer.predict_det.TextDetector(args)
        self.text_recognizer = paddleocr.tools.infer.predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = paddleocr.tools.infer.predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0
