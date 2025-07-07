import torch


def batchify(batch, tokenizer, max_length: int, use_trunc=False):
    """collate_fn
    Args:
        batch
        tokenizer
        max_length (int)
        use_trunc (bool)

    NOTE data["image"] can be None (e.g., text-instruction dataset)
    NOTE every batch for each device SHOULD have one image at least;
        if all batch data are text-only ones, error would occurs.
    """
    output_batch = {}
    image_list = [data["image"] for data in batch]

    num_images_per_sample = torch.LongTensor(
        [
            imgs["pixel_values"].shape[0] if imgs is not None else 0
            for imgs in image_list
        ]
    )

    # 1. remove None images from image_list
    image_list = [img for img in image_list if img is not None]

    # 2. collate for images: [num_images, c, h, w]
    images = [imgs["pixel_values"] for imgs in image_list]
    image_tensor = torch.cat(images) if images else None
    output_batch["pixel_values"] = image_tensor

    # 3. collate for text
    text_batch = [data["text"] for data in batch]
    padding = "longest" if use_trunc else "max_length"
    text_batch = tokenizer.batch_collate_pad(
        text_batch,
        padding=padding,
        padding_side="right",
        max_length=max_length,
    )

    # NOTE [bw-compat] Do not use attention mask for training, it will be generated automatically.
    text_batch.pop("attention_mask")

    output_batch.update(
        {
            **text_batch,
            "num_images": num_images_per_sample,
            "image_path": [data["image_path"] for data in batch],
        }
    )

    return output_batch
