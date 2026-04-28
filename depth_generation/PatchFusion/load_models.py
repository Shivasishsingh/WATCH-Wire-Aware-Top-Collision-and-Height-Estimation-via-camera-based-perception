from mmengine.config import Config
cfg_path = '/Vaibhav/shivasish1/sam2/PatchFusion/configs/patchfusion_depthanything/depthanything_vitl_patchfusion_u4k.py'
cfg = Config.fromfile(cfg_path) # load corresponding config for depth-anything vitl.
model = build_model(cfg.model) # build the model 
print(model.load_dict(torch.load(cfg.ckp_path)['model_state_dict']), logger='current') # load checkpoint