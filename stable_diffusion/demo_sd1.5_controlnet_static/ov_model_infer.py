from optimum.intel import OVStableDiffusionControlNetPipeline
from controlnet_aux import OpenposeDetector
from pathlib import Path
import numpy as np
import os
from PIL import Image
from diffusers import UniPCMultistepScheduler
import requests
import torch

NEED_STATIC = True
STATIC_SHAPE = [1024,1024]
DEVICE_NAME = "GPU.1"
if NEED_STATIC:
    print("Using static models")
    scheduler = UniPCMultistepScheduler.from_config("scheduler_config.json")
    ov_config ={"CACHE_DIR": "", 'INFERENCE_PRECISION_HINT': 'f16'}
    if not os.path.exists("ov_models_static"):
        if os.path.exists("ov_models_dynamic"):
            print("load dynamic models from local ov files and reshape to static")
            ov_pipe = OVStableDiffusionControlNetPipeline.from_pretrained(Path("ov_models_dynamic"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1])
            ov_pipe.reshape(batch_size=1 ,height=STATIC_SHAPE[0], width=STATIC_SHAPE[1], num_images_per_prompt=1)
            ov_pipe.save_pretrained(save_directory="./ov_models_static")
            print("Static model is saved in ./ov_models_static")  
        else:
            raise ValueError("No ov_models_dynamic exists, please trt ov_model_export.py first")
    else:
        print("load static models from local ov files")
        ov_pipe = OVStableDiffusionControlNetPipeline.from_pretrained(Path("ov_models_static"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1])
else:
    scheduler = UniPCMultistepScheduler.from_config("scheduler_config.json")
    ov_config ={"CACHE_DIR": "", 'INFERENCE_PRECISION_HINT': 'f16'}
    print("load dynamic models from local ov files")
    ov_pipe = OVStableDiffusionControlNetPipeline.from_pretrained(Path("ov_models_dynamic"), scheduler=scheduler, device=DEVICE_NAME, compile=True, ov_config=ov_config)

seed = 42
torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# load image for controlnet, or you can use your own image, or genereate image with Openpose Openvino model, 
# notice that Openpose model is not supported by OVStableDiffusionControlNetPipeline yet, so you need to convert it to openvino model first manually.

if os.path.exists("pose_1024.png"):
    pose = Image.open(Path("pose_1024.png"))
else:
    import torch
    import openvino as ov
    from collections import namedtuple
    print(f"pose.png not found, use openpose to generate pose image.")
    pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    example_url = "https://user-images.githubusercontent.com/29454499/224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg"
    #convert pytorch model of openpose to openvino
    class OpenPoseOVModel:
        """Helper wrapper for OpenPose model inference"""

        def __init__(self, core, model, device="AUTO"):
            self.core = core
            self.model = model
            self.compiled_model = core.compile_model(self.model, device)

        def __call__(self, input_tensor: torch.Tensor):
            """
            inference step

            Parameters:
            input_tensor (torch.Tensor): tensor with prerpcessed input image
            Returns:
            predicted keypoints heatmaps
            """
            h, w = input_tensor.shape[2:]
            input_shape = self.model.input(0).shape
            if h != input_shape[2] or w != input_shape[3]:
                self.reshape_model(h, w)
            results = self.compiled_model(input_tensor)
            return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])

        def reshape_model(self, height: int, width: int):
            """
            helper method for reshaping model to fit input data

            Parameters:
            height (int): input tensor height
            width (int): input tensor width
            Returns:
            None
            """
            self.model.reshape({0: [1, 3, height, width]})
            self.compiled_model = self.core.compile_model(self.model)

        def parameters(self):
            Device = namedtuple("Device", ["device"])
            return [Device(torch.device("cpu"))]
    
    with torch.no_grad():
        ov_model = ov.convert_model(
            pose_estimator.body_estimation.model,
            example_input=torch.zeros([1, 3, 184, 136]),
            input=[[1, 3, 184, 136]],
        )
    print(f"Converted openpose model to openvino")
    core = ov.Core() 
    ov_openpose = OpenPoseOVModel(core, ov_model, device="CPU")
    pose_estimator.body_estimation.model = ov_openpose  
    img = Image.open(requests.get(example_url, stream=True).raw)    
    pose = pose_estimator(img)
    pose = pose.resize((1024,1024))
    pose.save("pose_1024.png")


#set prompt, negative_prompt, num_inference_steps and image to get result
prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

result = ov_pipe(prompt=prompt, image=pose, num_inference_steps=20, negative_prompt=negative_prompt, height=STATIC_SHAPE[0], width=STATIC_SHAPE[1])

result[0].save("result_1024.png")