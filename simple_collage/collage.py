import torch
import torch.nn.functional as F
import pydiffvg 
from tqdm import tqdm
import cv2
import numpy as np
from outline_svg import outline_svg
import json
from utils.image_process import mask2targetimg,svg_to_img
from torchvision.transforms import ToPILImage
from utils.svg_process import init_diffvg
# ==================== 核心函数 ====================

def get_raw_control_points(shapes, centroids, device):
    """获取原始控制点（减去质心）"""
    raw_points_list = []
    for i, shape in enumerate(shapes):
        points = shape.points.to(device)
        centroid = torch.tensor(centroids[i], device=device, dtype=torch.float32)
        raw_points_list.append(points - centroid)
    return torch.stack(raw_points_list)


def weights_mse(mask, raster_img, target_img, scale=1):
    """加权MSE损失"""
    loss_mse = F.mse_loss(raster_img, target_img)
    mask_raster_img = mask * raster_img
    mask_target_img = mask * target_img
    loss_mse += F.mse_loss(mask_raster_img, mask_target_img) * 100
    return loss_mse * scale


def exclude_loss(raster_img, scale=1):
    """排斥损失"""
    img = F.relu(178/255 - raster_img)
    return torch.sum(img) * scale


def grid_based_sampling(contour, num_points, canvas_width, canvas_height):
    """
    基于网格的采样，通过收缩轮廓避免点在边缘初始化
    
    Args:
        contour: OpenCV轮廓
        num_points: 需要生成的点数
    
    Returns:
        生成的点列表
    """
    if contour is None or num_points <= 0:
        return []
    
    # 1. 获取轮廓边界框和面积
    x, y, w, h = cv2.boundingRect(contour)
    contour_area = cv2.contourArea(contour)
    if contour_area <= 0:
        return []
    
    # 2. 收缩轮廓
    # 创建二值图像
    binary_img = np.zeros((canvas_height, canvas_width), dtype=np.uint8) 
    cv2.drawContours(binary_img, [contour], -1, 255, -1)
    
    # 3. 动态调整网格间距，确保生成的点数 >= num_points
    grid_size = int(np.sqrt(contour_area / num_points))
    
    # 使用基于grid_size的形态学腐蚀操作收缩轮廓
    # 腐蚀核大小基于grid_size，确保腐蚀程度与网格密度匹配
    erosion_size = max(1, grid_size // 2)  # 腐蚀核大小为grid_size的一半，至少为1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size*2+1, erosion_size*2+1))
    shrunk_img = cv2.erode(binary_img, kernel, iterations=1)
    
    # 找到收缩后的轮廓
    shrunk_contours, _ = cv2.findContours(shrunk_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有找到收缩后的轮廓，使用原轮廓
    if not shrunk_contours:
        target_contour = contour
    else:
        # 直接取第一个（也是唯一的）收缩轮廓
        target_contour = shrunk_contours[0] 
    
    # 4. 动态调整网格间距，确保生成的点数 >= num_points
    points = []
    
    while True:
        points.clear()
        # 生成网格点
        for i in range(x, x + w, grid_size):
            for j in range(y, y + h, grid_size):
                if cv2.pointPolygonTest(target_contour, (i, j), False) >= 0:
                    points.append((i, j))
        
        # 检查是否生成足够的点
        if len(points) >= num_points:
            break
        else:
            # 缩小网格间距（例如每次减少10%）
            grid_size = max(1, int(grid_size * 0.9))  # 避免grid_size=0
    
    # 5. 随机下采样到目标点数（保证均匀性）
    if len(points) > num_points:
        # 计算轮廓质心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2  # 退化为边界框中心
        
        # 按点到质心的距离排序 
        points.sort(key=lambda p: np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2))
        
        # 保留最近的num_points个点 
        points = points[:num_points]
    
    return points

def main(outline_files, uniform_files, json_files, container_path, output_dir="./output", render_size=1000, num_iters=150,mark_num = 30):
    """主函数"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_diffvg(device)
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    outline_files = [outline_files] * mark_num
    uniform_files = [uniform_files] * mark_num
    json_files = [json_files] * mark_num

    #读取outline_files 
    shapes = [] 
    for i, outline_file in enumerate(outline_files):
        _, _, svg_shapes, _ = pydiffvg.svg_to_scene(outline_file)
        shapes.append(svg_shapes[0]) 
    #从json文件读取centroids
    centroids = []
    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as file:
            data = json.load(file)
            centroids.append([data['center_x'], data['center_y']])

    from PIL import Image
    import numpy as np
    # 获取原始控制点（减去质心）
    raw_control_points_tensor = get_raw_control_points(shapes, centroids, device)
    print(raw_control_points_tensor)
    raw_control_points_tensor.requires_grad = False
    
    # 使用grid_based_sampling初始化位置
    from PIL import Image
    container_img = Image.open(container_path).convert("L").resize((render_size, render_size))
    container_mask = np.array(container_img)
    black_mask = container_mask < 128  # 黑色区域为True
    binary_img = (black_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓（通常是主要的黑色区域）
    largest_contour = max(contours, key=cv2.contourArea)
    # 使用grid_based_sampling生成采样点
    sampled_points = grid_based_sampling(
        largest_contour, 
        num_points=mark_num, 
        canvas_width=render_size, 
        canvas_height=render_size
    )
    
    # 转换为位置列表 [x, y]
    chosen_positions = [[int(p[0]), int(p[1])] for p in sampled_points]
    
    # 转换为tensor (注意：坐标顺序是 [x, y])
    pos_data = np.array(chosen_positions, dtype=np.float32)
    pos_data = pos_data.reshape(mark_num, 1, 2)
    pos_tensor = torch.tensor(pos_data, dtype=torch.float32, device=device, requires_grad=True)
    
    # 创建固定的大小tensor（不优化，所有形状大小相同）
    size_tensor = torch.ones((mark_num, 1, 1), dtype=torch.float32, device=device, requires_grad=False)
    # 创建全局缩放因子（可优化）
    global_size = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
    angle_tensor = torch.zeros((mark_num, 1), dtype=torch.float32, device=device, requires_grad=True)
    
    # 创建形状组
    shape_groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.LongTensor([i]),
        fill_color=torch.FloatTensor([0, 0, 0, 0.3]),
        stroke_color=torch.FloatTensor([0, 0, 0, 0.3])
    ) for i in range(mark_num)]
    
    # 目标图像和掩码
    target_img, mask_img = mask2targetimg(container_path, device, render_size)
    
    # 创建保存目录
    import os
    os.makedirs(f"{output_dir}/iterations", exist_ok=True)
    
    # 优化器（只优化 global_size，不优化 size_tensor）
    optimizer = torch.optim.Adam([
        {'params': [global_size], 'lr': 0.1},
        {'params': [angle_tensor], 'lr': 0.05},
        {'params': [pos_tensor], 'lr': 1.0}
    ], betas=(0.9, 0.9), eps=1e-6)
    
    # 优化循环
    print("开始优化布局...")
    for epoch in tqdm(range(num_iters), desc="优化布局"):
        primitive_list1 = raw_control_points_tensor * size_tensor * global_size
        points_2 = torch.zeros_like(primitive_list1, device=device)
        points_2[:, :, 0] = primitive_list1[:, :, 0] * torch.cos(angle_tensor) - primitive_list1[:, :, 1] * torch.sin(angle_tensor)
        points_2[:, :, 1] = primitive_list1[:, :, 0] * torch.sin(angle_tensor) + primitive_list1[:, :, 1] * torch.cos(angle_tensor)
        points_2 = points_2 + pos_tensor
        
        # 更新形状点
        for i in range(mark_num):
            shapes[i].points = points_2[i]
        
        # 渲染图像
        raster_img = svg_to_img(shapes, shape_groups, render_size, render_size) 
        # 每10epoch保存一次rasterimg
        if (epoch + 1) % 10 == 0:
            image = ToPILImage()(raster_img.detach())
            image.save(f"{output_dir}/iterations/{epoch}_rasterimg.png") 
        # 计算损失
        loss_mse = weights_mse(mask_img, raster_img, target_img, scale=15e4)
        loss_exclude = exclude_loss(raster_img, scale=0.8)
        loss = loss_mse + loss_exclude
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每一轮都保存SVG
        pydiffvg.save_svg(f"{output_dir}/iterations/{epoch}.svg",
                         render_size,
                         render_size,
                         shapes,
                         shape_groups)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_iters}, Loss: {loss.item():.6f}, MSE: {loss_mse.item():.6f}, Exclude: {loss_exclude.item():.6f}")
    
    # 保存最终结果
    os.makedirs(f"{output_dir}/final", exist_ok=True)
    
    # 参考 post_svgs 函数，重新加载原始SVG并应用优化后的变换
    print("保存最终渲染结果...")
    shapes1 = []
    shape_groups1 = []
    count = 0
    
    # 将优化后的参数移到CPU并reshape
    size_tensor_cpu = size_tensor.detach().to(torch.device('cpu'))
    angle_tensor_cpu = angle_tensor.detach().to(torch.device('cpu'))
    pos_tensor_cpu = pos_tensor.detach().to(torch.device('cpu'))
    global_size_cpu = global_size.detach().to(torch.device('cpu'))
    
    
    # 重新加载原始SVG并应用变换
    for i in range(mark_num):
        _, _, shapes, shape_groups = pydiffvg.svg_to_scene(uniform_files[i])
        
        for j, shape in enumerate(shapes):
            with open(json_files[i], 'r') as file:
                data = json.load(file)
                center = torch.tensor([data["center_x"], data["center_y"]], dtype=torch.float32) 
            points_1 = (shape.points - center) * size_tensor_cpu[i, 0, 0] * global_size_cpu
            points_2 = torch.zeros_like(points_1)
            # angle_tensor_cpu 是 (mark_num, 1)，需要取 [i, 0]
            angle = angle_tensor_cpu[i, 0]
            points_2[:, 0] = points_1[:, 0] * torch.cos(angle) - points_1[:, 1] * torch.sin(angle)
            points_2[:, 1] = points_1[:, 0] * torch.sin(angle) + points_1[:, 1] * torch.cos(angle)
            # pos_tensor_cpu 是 (mark_num, 1, 2)，需要取 [i, 0, :]
            points_2 = points_2 + pos_tensor_cpu[i, 0, :]
            
            shape.points = points_2
            shapes1.append(shape)
            
            shape_groups[j].shape_ids = torch.LongTensor([count])
            count += 1
            shape_groups1.append(shape_groups[j])
    
    # 保存最终结果
    pydiffvg.save_svg(f"{output_dir}/final/final_result.svg",
                     render_size,
                     render_size,
                     shapes1,
                     shape_groups1)
    
    print(f"完成! 结果保存在 {output_dir}/final/final_result.svg")
    


if __name__ == "__main__": 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    outline_svg(device)
    outline_files = "./primitive/outline_files/outline_1.svg"
    uniform_files = "./primitive/outline_files/uniform_1.svg"
    json_files = "./primitive/outline_files/1.json"
    main( outline_files, uniform_files, json_files, "./container.png", "./output", 1000, 150,30)
