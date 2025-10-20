"""
测试插值操作是否会修改原始数据和掩码
Test whether interpolation operations modify original data and masks
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import copy

def test_bilinear_sampler_data_modification():
    """测试双线性采样器是否修改原始数据"""
    print("=== 测试双线性采样器数据修改 ===")
    
    # 创建测试数据
    original_img = torch.randn(1, 3, 64, 64, dtype=torch.float32)
    original_coords = torch.randn(1, 32, 32, 2, dtype=torch.float32) * 30  # 坐标范围
    
    # 保存原始数据的副本用于比较
    img_copy = original_img.clone()
    coords_copy = original_coords.clone()
    
    print(f"原始图像形状: {original_img.shape}")
    print(f"原始坐标形状: {original_coords.shape}")
    print(f"原始图像数据哈希: {hash(original_img.data_ptr())}")
    print(f"原始坐标数据哈希: {hash(original_coords.data_ptr())}")
    
    # 执行双线性采样
    H, W = original_img.shape[-2:]
    xgrid, ygrid = original_coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    
    # 执行插值操作
    sampled_img = F.grid_sample(original_img, grid, align_corners=True)
    
    # 检查原始数据是否被修改
    img_modified = not torch.equal(original_img, img_copy)
    coords_modified = not torch.equal(original_coords, coords_copy)
    
    print(f"插值后图像形状: {sampled_img.shape}")
    print(f"原始图像是否被修改: {img_modified}")
    print(f"原始坐标是否被修改: {coords_modified}")
    print(f"插值后图像数据哈希: {hash(sampled_img.data_ptr())}")
    
    # 验证数据完整性
    print(f"原始图像数据范围: [{original_img.min():.4f}, {original_img.max():.4f}]")
    print(f"插值后图像数据范围: [{sampled_img.min():.4f}, {sampled_img.max():.4f}]")
    
    return not img_modified and not coords_modified


def test_cv2_resize_mask_modification():
    """测试cv2.resize是否修改原始掩码"""
    print("\n=== 测试cv2.resize掩码修改 ===")
    
    # 创建测试掩码
    original_mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
    
    # 保存原始掩码的副本
    mask_copy = original_mask.copy()
    
    print(f"原始掩码形状: {original_mask.shape}")
    print(f"原始掩码数据类型: {original_mask.dtype}")
    print(f"原始掩码数据指针: {original_mask.data.tobytes()[:20]}")  # 前20字节
    print(f"原始掩码唯一值: {np.unique(original_mask)}")
    
    # 执行resize操作
    resized_mask = cv2.resize(original_mask, (50, 50), interpolation=cv2.INTER_NEAREST)
    
    # 检查原始掩码是否被修改
    mask_modified = not np.array_equal(original_mask, mask_copy)
    
    print(f"调整后掩码形状: {resized_mask.shape}")
    print(f"调整后掩码数据类型: {resized_mask.dtype}")
    print(f"原始掩码是否被修改: {mask_modified}")
    print(f"调整后掩码唯一值: {np.unique(resized_mask)}")
    
    # 测试布尔类型掩码
    bool_mask = original_mask.astype(bool)
    bool_mask_copy = bool_mask.copy()
    
    resized_bool_mask = cv2.resize(bool_mask.astype(np.uint8), 
                                  (75, 75), 
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
    
    bool_mask_modified = not np.array_equal(bool_mask, bool_mask_copy)
    
    print(f"布尔掩码是否被修改: {bool_mask_modified}")
    print(f"调整后布尔掩码唯一值: {np.unique(resized_bool_mask)}")
    
    return not mask_modified and not bool_mask_modified


def test_torch_interpolate_modification():
    """测试torch.nn.functional.interpolate是否修改原始数据"""
    print("\n=== 测试torch.nn.functional.interpolate数据修改 ===")
    
    # 创建测试数据
    original_tensor = torch.randn(1, 2, 32, 32, dtype=torch.float32)
    
    # 保存原始数据的副本
    tensor_copy = original_tensor.clone()
    
    print(f"原始张量形状: {original_tensor.shape}")
    print(f"原始张量数据类型: {original_tensor.dtype}")
    print(f"原始张量数据哈希: {hash(original_tensor.data_ptr())}")
    print(f"原始张量数据范围: [{original_tensor.min():.4f}, {original_tensor.max():.4f}]")
    
    # 执行插值操作
    upsampled_tensor = F.interpolate(original_tensor, scale_factor=8, mode='bilinear', align_corners=True)
    
    # 检查原始数据是否被修改
    tensor_modified = not torch.equal(original_tensor, tensor_copy)
    
    print(f"上采样后张量形状: {upsampled_tensor.shape}")
    print(f"原始张量是否被修改: {tensor_modified}")
    print(f"上采样后张量数据哈希: {hash(upsampled_tensor.data_ptr())}")
    print(f"上采样后张量数据范围: [{upsampled_tensor.min():.4f}, {upsampled_tensor.max():.4f}]")
    
    return not tensor_modified


def test_in_place_operations():
    """测试就地操作对数据的影响"""
    print("\n=== 测试就地操作 ===")
    
    # 测试numpy数组的就地操作
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    arr_copy = arr.copy()
    arr_id = id(arr)
    
    print(f"原始数组: {arr}")
    print(f"原始数组ID: {arr_id}")
    
    # 非就地操作
    arr_new = arr * 2
    print(f"非就地操作后原数组: {arr}")
    print(f"新数组: {arr_new}")
    print(f"原数组ID是否改变: {id(arr) != arr_id}")
    print(f"原数组是否被修改: {not np.array_equal(arr, arr_copy)}")
    
    # 就地操作
    arr *= 2
    print(f"就地操作后数组: {arr}")
    print(f"就地操作后ID是否改变: {id(arr) != arr_id}")
    print(f"就地操作是否修改原数组: {not np.array_equal(arr, arr_copy)}")
    
    # 测试torch张量的就地操作
    tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tensor_copy = tensor.clone()
    tensor_ptr = tensor.data_ptr()
    
    print(f"\n原始张量: {tensor}")
    print(f"原始张量数据指针: {tensor_ptr}")
    
    # 非就地操作
    tensor_new = tensor * 2
    print(f"非就地操作后原张量: {tensor}")
    print(f"新张量: {tensor_new}")
    print(f"原张量数据指针是否改变: {tensor.data_ptr() != tensor_ptr}")
    print(f"原张量是否被修改: {not torch.equal(tensor, tensor_copy)}")
    
    # 就地操作
    tensor *= 2
    print(f"就地操作后张量: {tensor}")
    print(f"就地操作后数据指针是否改变: {tensor.data_ptr() != tensor_ptr}")
    print(f"就地操作是否修改原张量: {not torch.equal(tensor, tensor_copy)}")
    
    return True


def test_interpolation_in_codebase_context():
    """测试代码库中具体插值操作的行为"""
    print("\n=== 测试代码库中插值操作 ===")
    
    # 模拟static_object_analyzer.py中的resize操作
    print("1. 测试static_object_analyzer中的resize操作:")
    
    # 创建测试数据，模拟edge_mask和refined_mask
    edge_mask = np.random.randint(0, 2, (100, 120), dtype=np.uint8).astype(bool)
    refined_mask = np.random.randint(0, 2, (100, 120), dtype=np.uint8).astype(bool)
    
    # 保存原始数据
    edge_mask_original = edge_mask.copy()
    refined_mask_original = refined_mask.copy()
    
    # 模拟代码中的操作
    target_height, target_width = 80, 100
    
    # 调整edge_mask (模拟第246-248行)
    if edge_mask.shape != (target_height, target_width):
        edge_mask_resized = cv2.resize(edge_mask.astype(np.uint8), 
                                     (target_width, target_height), 
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # 调整refined_mask (模拟第252-254行)
    if refined_mask.shape != (target_height, target_width):
        refined_mask_resized = cv2.resize(refined_mask.astype(np.uint8), 
                                        (target_width, target_height), 
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # 检查原始数据是否被修改
    edge_mask_modified = not np.array_equal(edge_mask, edge_mask_original)
    refined_mask_modified = not np.array_equal(refined_mask, refined_mask_original)
    
    print(f"   edge_mask原始形状: {edge_mask_original.shape}")
    print(f"   edge_mask调整后形状: {edge_mask_resized.shape}")
    print(f"   edge_mask是否被修改: {edge_mask_modified}")
    print(f"   refined_mask原始形状: {refined_mask_original.shape}")
    print(f"   refined_mask调整后形状: {refined_mask_resized.shape}")
    print(f"   refined_mask是否被修改: {refined_mask_modified}")
    
    # 模拟raft_model.py中的F.interpolate操作
    print("\n2. 测试raft_model中的F.interpolate操作:")
    
    # 创建测试数据，模拟coords1 - coords0
    coords_diff = torch.randn(1, 2, 16, 20, dtype=torch.float32)
    coords_diff_original = coords_diff.clone()
    
    # 模拟第330行的操作
    flow_up = 8 * F.interpolate(coords_diff, scale_factor=8, mode='bilinear', align_corners=True)
    
    # 检查原始数据是否被修改
    coords_diff_modified = not torch.equal(coords_diff, coords_diff_original)
    
    print(f"   coords_diff原始形状: {coords_diff_original.shape}")
    print(f"   flow_up形状: {flow_up.shape}")
    print(f"   coords_diff是否被修改: {coords_diff_modified}")
    print(f"   原始数据范围: [{coords_diff_original.min():.4f}, {coords_diff_original.max():.4f}]")
    print(f"   插值后数据范围: [{flow_up.min():.4f}, {flow_up.max():.4f}]")
    
    # 模拟bilinear_sampler函数
    print("\n3. 测试bilinear_sampler函数:")
    
    img = torch.randn(4, 256, 32, 40, dtype=torch.float32)
    coords = torch.randn(4, 64, 80, 2, dtype=torch.float32) * 20
    
    img_original = img.clone()
    coords_original = coords.clone()
    
    # 模拟bilinear_sampler函数的操作
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    
    sampled_img = F.grid_sample(img, grid, align_corners=True)
    
    img_modified = not torch.equal(img, img_original)
    coords_modified = not torch.equal(coords, coords_original)
    
    print(f"   img原始形状: {img_original.shape}")
    print(f"   sampled_img形状: {sampled_img.shape}")
    print(f"   img是否被修改: {img_modified}")
    print(f"   coords是否被修改: {coords_modified}")
    
    return not edge_mask_modified and not refined_mask_modified and not coords_diff_modified and not img_modified and not coords_modified


def run_all_interpolation_tests():
    """运行所有插值测试"""
    print("开始测试插值操作是否会修改原始数据和掩码...\n")
    
    tests = [
        test_bilinear_sampler_data_modification,
        test_cv2_resize_mask_modification,
        test_torch_interpolate_modification,
        test_in_place_operations,
        test_interpolation_in_codebase_context,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"测试 {test.__name__} 发生异常: {e}")
            results.append(False)
    
    # 总结
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== 插值测试总结 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有插值测试通过！")
        print("\n结论:")
        print("✓ 插值操作不会修改原始数据")
        print("✓ 插值操作不会修改原始掩码")
        print("✓ 插值操作是安全的，不会产生副作用")
    else:
        print(f"⚠ {total - passed} 个测试失败")
        print("需要进一步检查插值操作的实现")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_interpolation_tests()
    
    if success:
        print("\n📋 总结报告:")
        print("=" * 50)
        print("插值操作分析结果:")
        print("1. cv2.resize() - 不修改原始数据，返回新的数组")
        print("2. F.grid_sample() - 不修改原始张量，返回新的张量")
        print("3. F.interpolate() - 不修改原始张量，返回新的张量")
        print("4. 所有插值操作都是函数式的，遵循不可变性原则")
        print("5. 原始数据和掩码在插值后保持完整和不变")
        print("=" * 50)
    else:
        print("\n需要修复插值操作中的问题")