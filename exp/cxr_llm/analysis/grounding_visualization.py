import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def norm2origin_box(box, width, height, resize_type):
    if resize_type == "shortest_edge":
        min_length = min(width, height)
        if height >= width:
            half_diff = (height - min_length) / 2
            x1 = box[0] * min_length / 100
            x2 = box[2] * min_length / 100
            y1 = box[1] * min_length / 100 + half_diff
            y2 = box[3] * min_length / 100 + half_diff
        else:
            half_diff = (width - min_length) / 2
            x1 = box[0] * min_length / 100 + half_diff
            x2 = box[2] * min_length / 100 + half_diff
            y1 = box[1] * min_length / 100
            y2 = box[3] * min_length / 100

    else:
        NotImplementedError
    return [round(x1), round(y1), round(x2), round(y2)]


def grounding_visualization(
    image_path,
    save_dir,
    resize_type,
    line_size,
    gt_box_list=[],
    predict_box_list=[],
    figsize=(10, 8),
):

    image_name = "visualization" + "_" + os.path.basename(image_path)

    image = Image.open(image_path).convert("RGB")
    original_size = (image.height, image.width)

    height = original_size[0]
    width = original_size[1]

    draw = ImageDraw.Draw(image)

    for gt_box_str in gt_box_list:
        gt_box = norm2origin_box(gt_box_str, width, height, resize_type)
        draw.rectangle(gt_box, outline="red", width=line_size)

    for predict_box_str in predict_box_list:
        predict_box = norm2origin_box(predict_box_str, width, height, resize_type)
        draw.rectangle(predict_box, outline="blue", width=line_size)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")

    image.save(os.path.join(save_dir, image_name))
    return gt_box, predict_box


if __name__ == "__main__":
    data_root = "datasets"
    image_path = os.path.join(
        data_root,
        "MIMIC-CXR/images/dff1e4b1-19095040-20f3ad74-e13d58a4-4603b22d.jpg",
    )

    save_dir = "visualization"
    os.makedirs(save_dir, exist_ok=True)

    """
    example
    """
    gt_box_list = [
        [10, 20, 30, 40],
        [15, 45, 35, 55],
        [50, 60, 70, 80],
        [95, 95, 100, 100],
    ]
    predict_box_list = [
        [0, 0, 5, 5],
        [15, 32, 39, 60],
    ]

    original_bbox = grounding_visualization(
        image_path,
        save_dir,
        gt_box_list=gt_box_list,
        predict_box_list=predict_box_list,
        resize_type="shortest_edge",
        line_size=6,
    )
