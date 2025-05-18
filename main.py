import cv2
import numpy as np
import os
import time

from audio_grid import PyoGridAudioManager
from yolov7_api import YOLOv7Wrapper


def compute_grid_homography(corner_points):
    assert len(corner_points) == 4, "Need exactly 4 corner points"
    dst_pts = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
    src_pts = np.array(corner_points, dtype=np.float32)
    H, status = cv2.findHomography(src_pts, dst_pts)
    return H


def map_points_to_grid_cells2(H, image_points):
    image_points_np = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    grid_points = cv2.perspectiveTransform(image_points_np, H).reshape(-1, 2)
    results = []
    for gx, gy in grid_points:
        i, j = int(gx), int(gy)
        if 0 <= i < 8 and 0 <= j < 8:
            results.append((i, j))
        else:
            results.append(None)
    return results

def map_points_to_grid_cells(H, image_points):
    if not image_points:            # no points this frame
        return []                   # avoids one extra transform call
    pts = np.array(image_points, dtype=np.float32).reshape(-1, 1, 2)
    grid = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    results = []
    for gx, gy in grid:
        i, j = int(np.floor(gx)), int(np.floor(gy))
        if 0 <= i < 8 and 0 <= j < 8:
            results.append((i, j))
        else:
            results.append(None)
    return results


def draw_grid_overlay(frame, H_inv, active_cells=None):
    for i in range(8):
        for j in range(8):
            cell = np.array([
                [[i, j]], [[i + 1, j]],
                [[i + 1, j + 1]], [[i, j + 1]]
            ], dtype=np.float32)
            pts = cv2.perspectiveTransform(cell, H_inv).astype(int).reshape(-1, 2)
            if active_cells and (i, j) in active_cells:
                cv2.fillPoly(frame, [pts], (0, 255, 0))
            else:
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=1)


def ask_user_for_4_points(image):
    clicked_points = []

    def mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((x, y))
            print(f"Clicked: ({x}, {y})")

    cv2.namedWindow("Select 4 grid corners")
    cv2.setMouseCallback("Select 4 grid corners", mouse_click)

    while True:
        disp = image.copy()
        for pt in clicked_points:
            cv2.circle(disp, pt, 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 grid corners", disp)
        key = cv2.waitKey(1)
        if key == 27:
            exit(1)
        if len(clicked_points) == 4:
            break

    cv2.destroyWindow("Select 4 grid corners")
    return clicked_points


def visualize_frame(frame, bboxes, bottom_points, cell_mappings, H_inv):
    active_cells = set(cell for cell in cell_mappings if cell is not None)
    draw_grid_overlay(frame, H_inv, active_cells)

    for (x, y, w, h) in bboxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for pt, cell in zip(bottom_points, cell_mappings):
        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        label = f"{cell}" if cell else "X"
        cv2.putText(frame, label, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


if __name__ == "__main__":
    # Use camera instead of video file
    camera_id = 0  # Usually 0 is the default camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: cannot open camera {camera_id}")
        exit(1)

    # Set camera properties if needed
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, initial_frame = cap.read()
    if not ret:
        print("Error: failed to read first frame from camera.")
        exit(1)

    corner_pts = ask_user_for_4_points(initial_frame)
    H = compute_grid_homography(corner_pts)
    H_inv = np.linalg.inv(H)

    yolo = YOLOv7Wrapper(weights='yolov7-tiny.pt', img_size=640, conf_thres=0.25, device='cpu')
    audio = PyoGridAudioManager("loops")
    previous_cells = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from camera.")
                break

            t1 = time.time()
            bboxes = yolo.infer(frame)
            t2 = time.time()
            print(f"Inference time: {t2 - t1:.3f}s")

            bottom_points = []
            for (x, y, w, h) in bboxes:
                for dx in range(0, w + 1, 10):
                    pt = (x + dx, y + h)
                    bottom_points.append(pt)

            cell_mappings = map_points_to_grid_cells(H, bottom_points)

            # Extract unique active cells (for audio logic only)
            active_cells = set(cell for cell in cell_mappings if cell is not None)

            # Update audio grid
            newly_on = active_cells - previous_cells
            newly_off = previous_cells - active_cells
            for cell in newly_on:
                audio.set_cell_state(*cell, on=True)
            for cell in newly_off:
                audio.set_cell_state(*cell, on=False)
            previous_cells = active_cells

            # Visualize full frame with exact point-to-cell mapping
            visualize_frame(frame, bboxes, bottom_points, cell_mappings, H_inv)

            cv2.imshow("Grid Visualization", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    finally:
        cap.release()
        audio.shutdown()
        cv2.destroyAllWindows()
