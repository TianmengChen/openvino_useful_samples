from optimum.intel import OVStableDiffusionControlNetPipeline
import os
from diffusers import UniPCMultistepScheduler

SD15_PYTORCH_MODEL_DIR="stable-diffusion-v1-5"
CONTROLNET_PYTORCH_MODEL_DIR="control_v11p_sd15_openpose"


if os.path.exists(SD15_PYTORCH_MODEL_DIR) and os.path.exists(CONTROLNET_PYTORCH_MODEL_DIR):
    scheduler = UniPCMultistepScheduler.from_config("scheduler_config.json")
    ov_pipe = OVStableDiffusionControlNetPipeline.from_pretrained(SD15_PYTORCH_MODEL_DIR, controlnet_model_id=CONTROLNET_PYTORCH_MODEL_DIR, compile=False, export=True, scheduler=scheduler,device="GPU.1")
    ov_pipe.save_pretrained(save_directory="./ov_models_dynamic")
    print("Dynamic model is saved in ./ov_models_dynamic")  

else:
    scheduler = UniPCMultistepScheduler.from_config("scheduler_config.json")
    ov_pipe = OVStableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet_model_id="lllyasviel/control_v11p_sd15_openpose", compile=False, export=True, scheduler=scheduler, device="GPU.1")
    ov_pipe.save_pretrained(save_directory="./ov_models_dynamic")
    print("Dynamic model is saved in ./ov_models_dynamic")


