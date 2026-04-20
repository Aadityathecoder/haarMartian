import cv2
import numpy as np
import os


class MartianDetector:
    def __init__(self, template_path, threshold=0.55):
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template image not found: {template_path}")

        self.threshold = threshold

        self.template = cv2.imread(template_path)
        if self.template is None:
            raise ValueError(f"Could not read template image: {template_path}")

        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

    def detect(self, frame):
        """
        Detects the Martian image in a camera frame.

        Returns:
            found: bool
            boxes: list of tuples (x, y, w, h, score)
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        boxes = []

        # Multi-scale matching so detection works if the image appears
        # bigger or smaller in the camera view.
        for scale in np.linspace(0.2, 1.5, 22):
            resized_template = cv2.resize(
                self.template_gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

            th, tw = resized_template.shape[:2]

            if th < 20 or tw < 20:
                continue

            if th >= frame_h or tw >= frame_w:
                continue

            result = cv2.matchTemplate(
                frame_gray,
                resized_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= self.threshold:
                x, y = max_loc
                boxes.append((x, y, tw, th, float(max_val)))

        boxes = self._non_max_suppression(boxes)
        return len(boxes) > 0, boxes

    def draw_detections(self, frame, boxes):
        for x, y, w, h, score in boxes:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                3
            )

            cv2.putText(
                frame,
                f"MARTIAN {score:.2f}",
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        if boxes:
            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 60),
                (0, 0, 180),
                -1
            )

            cv2.putText(
                frame,
                "MARTIAN DETECTED!!!",
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (255, 255, 255),
                3
            )

        return frame

    def _non_max_suppression(self, boxes, overlap_threshold=0.35):
        if not boxes:
            return []

        boxes_np = np.array(boxes, dtype=np.float32)

        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 0] + boxes_np[:, 2]
        y2 = boxes_np[:, 1] + boxes_np[:, 3]
        scores = boxes_np[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / areas[order[1:]]

            order = order[np.where(overlap <= overlap_threshold)[0] + 1]

        return [tuple(boxes[int(i)]) for i in keep]

