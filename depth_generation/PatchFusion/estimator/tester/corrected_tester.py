import os
import cv2
import wandb
import numpy as np
import torch
import mmengine
from mmengine.optim import build_optim_wrapper
import torch.optim as optim
import matplotlib.pyplot as plt
from mmengine.dist import get_dist_info, collect_results_cpu, collect_results_gpu
from mmengine import print_log
from estimator.utils import colorize, colorize_infer_pfv1, colorize_rescale
import torch.nn.functional as F
from tqdm import tqdm
from mmengine.utils import mkdir_or_exist
import copy
from skimage import io
import kornia
from PIL import Image

class Tester:
    """
    Tester class
    """
    def __init__(
        self, 
        config,
        runner_info,
        dataloader,
        model):
       
        self.config = config
        self.runner_info = runner_info
        self.dataloader = dataloader
        self.model = model
        self.collect_input_args = config.collect_input_args
    
    def collect_input(self, batch_data):
        collect_batch_data = dict()
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                if k in self.collect_input_args:
                    collect_batch_data[k] = v.cuda()
        return collect_batch_data
    
    @torch.no_grad()
    def run(self, cai_mode='p16', process_num=4, image_raw_shape=[2160, 3840], patch_split_num=[4, 4]):
        
        results = []
        dataset = self.dataloader.dataset
        loader_indices = self.dataloader.batch_sampler
        
        rank, world_size = get_dist_info()
        if self.runner_info.rank == 0:
            prog_bar = mmengine.utils.ProgressBar(len(dataset))

        for idx, (batch_indices, batch_data) in enumerate(zip(loader_indices, self.dataloader)):
            
            batch_data_collect = self.collect_input(batch_data)
            
            tile_cfg = dict()
            tile_cfg['image_raw_shape'] = image_raw_shape
            tile_cfg['patch_split_num'] = patch_split_num # use a customized value instead of the default [4, 4] for 4K images
            result, log_dict = self.model(mode='infer', cai_mode=cai_mode, process_num=process_num, tile_cfg=tile_cfg, **batch_data_collect) # might use test/val to split cases
            
            if self.runner_info.save:
                # Get original image path and load to get dimensions
                img_file_basename = batch_data['img_file_basename'][0]
                
                # Try to get original image path from dataset
                rgb_image_dir = dataset.rgb_image_dir if hasattr(dataset, 'rgb_image_dir') else None
                
                if rgb_image_dir is not None:
                    # Load original image to get its dimensions
                    img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                    original_img_path = None
                    
                    for ext in img_extensions:
                        potential_path = os.path.join(rgb_image_dir, img_file_basename + ext)
                        if os.path.exists(potential_path):
                            original_img_path = potential_path
                            break
                    
                    if original_img_path is not None:
                        # Read original image to get its shape
                        original_img = cv2.imread(original_img_path)
                        if original_img is not None:
                            original_h, original_w = original_img.shape[:2]
                            
                            # Resize result back to original image dimensions
                            if result.dim() == 2:
                                result_to_resize = result.unsqueeze(0).unsqueeze(0)
                            elif result.dim() == 3:
                                result_to_resize = result.unsqueeze(0)
                            else:
                                result_to_resize = result
                            
                            # Resize depth output to original image dimensions
                            result_resized = F.interpolate(
                                result_to_resize,
                                size=(original_h, original_w),
                                mode='bilinear',
                                align_corners=False
                            ).squeeze()
                        else:
                            result_resized = result
                    else:
                        result_resized = result
                else:
                    result_resized = result
                
                # Save colorized depth map at original resolution
                if self.runner_info.gray_scale:
                    color_pred = colorize(result_resized, cmap='gray_r')[:, :, [2, 1, 0]]
                else:
                    color_pred = colorize(result_resized, cmap='magma_r')[:, :, [2, 1, 0]]
                cv2.imwrite(os.path.join(self.runner_info.work_dir, '{}.png'.format(img_file_basename)), color_pred)
            
                # Save as 16-bit PNG at original resolution
                raw_depth = Image.fromarray((result_resized.clone().squeeze().detach().cpu().numpy()*256).astype('uint16'))
                raw_depth.save(os.path.join(self.runner_info.work_dir, '{}_uint16.png'.format(img_file_basename)))

            if batch_data_collect.get('depth_gt', None) is not None:
                metrics = dataset.get_metrics(
                    batch_data_collect['depth_gt'], 
                    result, 
                    seg_image=batch_data_collect.get('seg_image', None),
                    disp_gt_edges=batch_data.get('boundary', None), 
                    image_hr=batch_data.get('image_hr', None))
                results.extend([metrics])
            
            if self.runner_info.rank == 0:
                batch_size = len(result) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
        
        if batch_data_collect.get('depth_gt', None) is not None:   
            results = collect_results_gpu(results, len(dataset))
            if self.runner_info.rank == 0:
                ret_dict = dataset.evaluate(results)
