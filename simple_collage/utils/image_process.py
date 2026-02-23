from torchvision import transforms
import torch
import pydiffvg
import numpy as np
from PIL import Image
def mask2targetimg(mask_dir,device,img_size=1000,is_binary=True):
    # 目标图像
    image = Image.open(mask_dir).convert("L").resize([img_size,img_size])
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    if is_binary:
        for x in range(width):
            for y in range(height):
                gray_value = image.getpixel((x, y))
                
                # RGB值设置为(178, 178, 178)
                if gray_value < 128:
                    rgb_image.putpixel((x, y), (0, 0, 0))
                else:
                    rgb_image.putpixel((x, y), (255, 255, 255))
    target_img = transforms.ToTensor()(rgb_image)
    target_img = target_img.to(device)

    # 掩码图像
    binary_image = np.array(image) > 128
    binary_image = binary_image.astype(int)
    mask_img = torch.tensor(binary_image,device=device,dtype=torch.float32,requires_grad=False)

    return target_img,mask_img

# 可微分的渲染svg
def svg_to_img(shapes, shape_groups, width, height):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
    width, height, shapes, shape_groups
    )
    _render = pydiffvg.RenderFunction.apply
    img = _render(width,  # width
                height,  # height
                2,  # num_samples_x
                2,  # num_samples_y
                0,  # seed
                None,
                *scene_args)
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=img.device)
    img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
    img = img.permute(2, 0, 1)
    return img

