import torch
import pydiffvg
from utils.image_process import svg_to_img
from torchvision.transforms import ToPILImage
import torch.nn.functional as F 
from torchvision import transforms
import cv2
import os
from scipy.ndimage import center_of_mass 
from utils.svg_process import init_diffvg
from tqdm import tqdm
import glob
from natsort import natsorted
import json
import numpy as np


def insert_point_to_longest_segment(points,num_segments=20):
    def distance(point1, point2):
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    # 找到最长线段
    max_length = 0
    longest_segment = None
    for i in range(len(points) - 1):
        dist = distance(points[i], points[i + 1])
        if dist > max_length:
            max_length = dist
            longest_segment = (i, i + 1)
    
    if longest_segment is None:
        return points
    
    # 计算最长线段的中心点
    start_idx, end_idx = longest_segment
    start_point = points[start_idx]
    end_point = points[end_idx]
    center_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    
    # 插入中心点到最长线段的位置
    new_points = np.insert(points, end_idx, center_point, axis=0)
    
    # 如果需要8个点，继续插入新点
    if len(new_points) == num_segments:
        return new_points
    # 递归调用直到插入到8个点
    return insert_point_to_longest_segment(new_points,num_segments)

# 在每个线段中插入新点
def insert_points_in_segments(points, num_interpolations=2):
    # 线性插值函数
    def interpolate_points(point1, point2, num_interpolations):
        # 生成 num_interpolations 个插入点
        x_values = np.linspace(point1[0], point2[0], num=num_interpolations + 2)[1:-1]
        y_values = np.linspace(point1[1], point2[1], num=num_interpolations + 2)[1:-1]
        interpolated_points = np.column_stack((x_values, y_values))
        return interpolated_points
    new_points = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i + 1]
        # 将原始点加入新点集
        new_points.append(point1)
        # 在当前线段中插入新点
        interpolated_points = interpolate_points(point1, point2, num_interpolations)
        new_points.extend(interpolated_points)
    # 将最后一个原始点加入新点集
    new_points.append(points[-1])
    
    return np.array(new_points)

def uniform_size(svg_dir,save_dir,fixed_size=80,margin=10,index=0):
    _, _, shapes, shape_groups = pydiffvg.svg_to_scene(svg_dir)
    shape = shapes[0]
    min_values, _ = torch.min(shape.points, dim=0)
    max_values, _ = torch.max(shape.points, dim=0)
    min_x,min_y = min_values[0],min_values[1]
    max_x,max_y = max_values[0],max_values[1]
    for shape in shapes:
        min_values, _ = torch.min(shape.points, dim=0)
        max_values, _ = torch.max(shape.points, dim=0)
        min_x =  min_values[0] if min_values[0] < min_x else min_x
        min_y = min_values[1] if min_values[1] < min_y else min_y
        max_x =  max_values[0] if max_values[0] > max_x else max_x
        max_y = max_values[1] if max_values[1] > max_y else max_y
    # fixed_size = 80 
    scaling_ratio = fixed_size/max(max_x-min_x,max_y-min_y)
    for shape in shapes:
        shape.points[:,0] = shape.points[:,0]-min_x
        shape.points[:,1] = shape.points[:,1]-min_y
        shape.points = shape.points*scaling_ratio
        shape.points[:,0] = shape.points[:,0]+margin
        shape.points[:,1] = shape.points[:,1]+margin

    pydiffvg.save_svg(f"{save_dir}/uniform_{index}.svg",
                        100,
                        100,
                        shapes,
                        shape_groups)
    return shapes

def get_target_img(shapes,img_size=100):
    path_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                                shape_ids=torch.LongTensor([i]),
                                fill_color=torch.FloatTensor([0,0,0,1]),
                                stroke_color=torch.FloatTensor([0,0,0,1])
                            )
        path_groups.append(path_group)

    target_img =  svg_to_img(shapes,path_groups,img_size,img_size)
    image = ToPILImage()(target_img.detach())
    return image

def init_shapes(image,num_segments=20):
    binary_image = np.array(image.convert('L'))
    binary_image = binary_image < 128
    binary_image = np.uint8(binary_image) * 255
    # 查找二值图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    kernel_size=1
    while len(contours)>1:
        kernel_size+=1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # 5x5的矩形核
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]

    simplified_contour = [1]*100
    epsilon = 0
    while len(simplified_contour)>num_segments:
        epsilon+=1
        # 进行轮廓近似，根据不同的阈值进行简化
        simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(simplified_contour) == num_segments:
        simplified_contour = simplified_contour[:,0,:]
    elif len(simplified_contour)<=num_segments-1:
        simplified_contour = insert_point_to_longest_segment(simplified_contour[:,0,:],num_segments)
    
    points = simplified_contour
    points = np.vstack((points, points[0]))
    points = insert_points_in_segments(points, num_interpolations=2)
    points = points[:-1]
    points = torch.FloatTensor(points)
    num_control_points = [2] * num_segments
    path = pydiffvg.Path(
                        num_control_points=torch.LongTensor(num_control_points),
                        points=points,
                        stroke_width=torch.tensor(0.0),
                        is_closed=True
                    )
    return path

def svg_optimize_img(img_size, shapes, shape_groups,image_taregt,device,save_svg_path,num_iters=1000,svg_i=0):
    image_taregt = image_taregt.convert('RGB') 
    image_taregt = transforms.ToTensor()(image_taregt)
    image_taregt = image_taregt.to(device)

    # 设置优化器
    params = {}
    points_vars = []
    for i, path in enumerate(shapes):
        path.id = i  # set point id
        path.points.requires_grad = True
        points_vars.append(path.points)

    params = {}
    params['point'] = points_vars
    lr_base = {
        "point": 1,
    }
    learnable_params = [
        {'params': params[ki], 'lr': lr_base[ki], '_id': str(ki)} for ki in sorted(params.keys())
    ]
    svg_optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)
    with tqdm(total=num_iters, desc="Processing value", unit="value") as pbar:
        for i in range(num_iters):
            img = svg_to_img(shapes, shape_groups,img_size,img_size)
            image_loss = F.mse_loss(img, image_taregt)
            loss = image_loss
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pbar.update(1)
            # pbar.set_description(f"epoch:{i}")
    pydiffvg.save_svg(f"{save_svg_path}/outline_{svg_i}.svg",
                    img_size,
                    img_size,
                    shapes,
                    shape_groups)

def get_area_centroid(target_img,save_svg_path,svg_i):
    # 打开图像并转换为灰度图
    image = target_img.convert('L')
    
    # 将灰度图像转换为二值图像（0 或 1）
    binary_image = np.array(image) < 128
    area = np.sum(binary_image == True)
    # 计算质心
    centroid = center_of_mass(binary_image)
    data = {
        "center_x": int(centroid[1]),
        "center_y": int(centroid[0]),
        "area": int(area),
    }
    with open(f"{save_svg_path}/{svg_i}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

def outline_svg(device):
    svg_files = glob.glob(os.path.join(f"./primitive/images", '*.svg'))
    primitive_files = natsorted(svg_files, key=lambda x: os.path.basename(x).lower())
    os.makedirs(f"./primitive/outline_files", exist_ok=True)

    with tqdm(total=len(primitive_files), desc="Processing value", unit="value") as pbar:
        for i,primitive_file in enumerate(primitive_files): 

            svg_normalized_size = 100
            shapes = uniform_size(primitive_file,fixed_size=svg_normalized_size*0.8,margin=svg_normalized_size*0.1,save_dir=f"./primitive/outline_files",index=i+1)
            target_img = get_target_img(shapes) 

            path = init_shapes(target_img,num_segments=20)
            path_group = pydiffvg.ShapeGroup(
                                    shape_ids=torch.LongTensor([0]),
                                    fill_color=torch.FloatTensor([0,0,0,1]),
                                    stroke_color=torch.FloatTensor([0,0,0,1])
                                )
            svg_optimize_img(svg_normalized_size,[path],[path_group],target_img,device,save_svg_path=f"./primitive/outline_files",num_iters=10,svg_i=i+1)
            get_area_centroid(target_img,f"./primitive/outline_files",svg_i=i+1)
            pbar.update(1)
            pbar.set_description(f"Number of primitives:{len(primitive_files)}")

# if __name__ == '__main__':
#     device = torch.device("cuda:0")
#     init_diffvg(device)
#     outline_svg(device)
