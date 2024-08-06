from pathlib import Path
import openvino as ov

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTokenizer
from diffusers import EulerDiscreteScheduler
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image

import PIL.Image
import torch
import numpy as np
import cv2
import os
import time

NEED_STATIC = True
STATIC_SHAPE = [1024,1024]

DEVICE_NAME="GPU.1"
COMPILE_CONFIG_FP32 = {'INFERENCE_PRECISION_HINT': 'f32', }
COMPILE_CONFIG_FP16 = {'INFERENCE_PRECISION_HINT': 'f16', }
UNET_OV_PATH = Path("./models_ov_dynamic/unet/openvino_model.xml")
CONTROLNET_OV_PATH = Path("./models_ov_dynamic/controlnet/openvino_model.xml")
TEXT_ENCODER_OV_PATH = Path("./models_ov_dynamic/encoder/openvino_model.xml")
TEXT_ENCODER_2_OV_PATH = Path("./models_ov_dynamic/encoder_2/openvino_model.xml")
TOKENIZER_OV_PATH = Path("./models_ov_dynamic/tokenizer")
TOKENIZER_2_OV_PATH = Path("./models_ov_dynamic/tokenizer_2")
SCHEDULER_OV_PATH = Path("./models_ov_dynamic/scheduler")
VAE_DECODER_OV_PATH = Path("./models_ov_dynamic/vae_decoder/openvino_model.xml")

UNET_STATIC_OV_PATH = Path("./models_ov_static/unet/openvino_model.xml")
CONTROLNET_STATIC_OV_PATH = Path("./models_ov_static/controlnet/openvino_model.xml")
TEXT_ENCODER_STATIC_OV_PATH = Path("./models_ov_static/encoder/openvino_model.xml")
TEXT_ENCODER_STATIC_2_OV_PATH = Path("./models_ov_static/encoder_2/openvino_model.xml")
VAE_DECODER_STATIC_OV_PATH = Path("./models_ov_static/vae_decoder/openvino_model.xml")

core = ov.Core()

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


if UNET_OV_PATH.exists() and CONTROLNET_OV_PATH.exists() and TEXT_ENCODER_OV_PATH.exists() and TEXT_ENCODER_2_OV_PATH.exists() and VAE_DECODER_OV_PATH.exists():
    print("Loading OpenVINO models")
else:
    controlnet = ControlNetModel.from_pretrained("./models_torch/mistoLine",torch_dtype=torch.float32,variant="fp16")
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained("./models_torch/stable-diffusion-xl-base-1.0", vae=vae, controlnet=controlnet)

#config save
if not TOKENIZER_OV_PATH.exists():
    pipe.tokenizer.save_pretrained(TOKENIZER_OV_PATH)

if not TOKENIZER_2_OV_PATH.exists():
    pipe.tokenizer_2.save_pretrained(TOKENIZER_2_OV_PATH)

if not SCHEDULER_OV_PATH.exists():
    pipe.scheduler.save_config(SCHEDULER_OV_PATH)
    
#controlnet
class ControlnetWrapper(torch.nn.Module):
    def __init__(
            self, 
            controlnet,
            sample_dtype=torch.float32,
            timestep_dtype=torch.int64,
            encoder_hidden_states_dtype=torch.float32,
            controlnet_cond_dtype=torch.float32,
            text_embeds_dtype=torch.float32,
            time_ids_dtype=torch.int64,
        ):
        super().__init__()
        self.controlnet = controlnet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states_dtype
        self.controlnet_cond_dtype = controlnet_cond_dtype
        self.text_embeds_dtype = text_embeds_dtype
        self.time_ids_dtype = time_ids_dtype

    def forward(
            self, 
            sample: torch.Tensor, 
            timestep: torch.Tensor, 
            encoder_hidden_states: torch.Tensor, 
            controlnet_cond: torch.Tensor, 
            text_embeds: torch.Tensor, 
            time_ids: torch.Tensor, 
        ):
            sample.to(self.sample_dtype)
            timestep.to(self.timestep_dtype)
            encoder_hidden_states.to(self.encoder_hidden_states_dtype)
            controlnet_cond.to(self.controlnet_cond_dtype)
            text_embeds.to(self.text_embeds_dtype)
            time_ids.to(self.time_ids_dtype)
            added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
            return self.controlnet(
            sample=sample, 
            timestep=timestep, 
            encoder_hidden_states=encoder_hidden_states, 
            controlnet_cond=controlnet_cond, 
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            )
    
inputs = {
    "sample": torch.randn((2, 4, 128, 128)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2, 77, 2048)),
    "controlnet_cond": torch.randn((2, 3, 1024, 1024)),
    "text_embeds": torch.randn((2, 1280)),
    "time_ids": torch.randn((2, 6)),
}

input_info = []
for name, inp in inputs.items():
    shape = ov.PartialShape(inp.shape)  
    # element_type = dtype_mapping[input_tensor.dtype]
    if len(shape) == 4:
        shape[0] = -1
        shape[2] = -1
        shape[3] = -1
    elif len(shape) == 3:
        shape[0] = -1
        shape[1] = -1
    elif len(shape) == 2:
        shape[0] = -1
        shape[1] = -1
    input_info.append((shape))

 


if not CONTROLNET_OV_PATH.exists():

    controlnet=ControlnetWrapper(controlnet)
    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs)
        ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
        ov.save_model(ov_model, CONTROLNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
    print("ControlNet successfully converted to IR")
    del controlnet
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")




#unet
dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}


class UnetWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        sample_dtype=torch.float32,
        timestep_dtype=torch.int64,
        encoder_hidden_states_dtype=torch.float32,
        text_embeds_dtype=torch.float32,
        time_ids_dtype=torch.int64,
        down_block_additional_residuals_dtype=torch.float32,
        mid_block_additional_residual_dtype=torch.float32,

    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states_dtype
        self.text_embeds_dtype = text_embeds_dtype
        self.time_ids_dtype = time_ids_dtype
        self.down_block_additional_residuals_dtype = down_block_additional_residuals_dtype
        self.mid_block_additional_residual_dtype = mid_block_additional_residual_dtype

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor, 
        time_ids: torch.Tensor, 
        down_block_additional_residuals: Tuple[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        text_embeds.to(self.text_embeds_dtype)
        time_ids.to(self.time_ids_dtype)
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
        )


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


if not UNET_OV_PATH.exists():
    inputs.pop("controlnet_cond", None)
    
    inputs["down_block_additional_residuals"] = down_block_res_samples
    inputs["mid_block_additional_residual"] = mid_block_res_sample

    unet = UnetWrapper(pipe.unet)
    unet.eval()

    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=inputs)
  
    flatten_inputs = flattenize_inputs(inputs.values())
    a = 1

    for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
        r_name = input_tensor.get_node().get_friendly_name()
        r_shape = ov.PartialShape(input_data.shape)
        print("============")
        print(r_name, r_shape)
        
        if len(r_shape) == 4:
            r_shape[0] = -1
            r_shape[2] = -1
            r_shape[3] = -1
        elif len(r_shape) == 3:
            r_shape[0] = -1
            r_shape[1] = -1
        elif len(r_shape) == 2:
            r_shape[0] = -1
            r_shape[1] = -1
        tn = "down_block_additional_residual_"
        if r_name not in ["sample", "timestep", "encoder_hidden_states", "mid_block_additional_residual", "text_embeds", "time_ids"] and len(r_shape)==4:
            n_name = tn + str(a)
            if a == 17:
                n_name = "down_block_additional_residual"
            input_tensor.get_node().set_friendly_name(n_name)
            a = a + 2
        print(input_tensor.get_node().get_friendly_name(), r_shape)
        print("============")
        input_tensor.get_node().set_partial_shape(r_shape)
        input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, UNET_OV_PATH)
    del ov_model
    cleanup_torchscript_cache()
    del unet
    del pipe.unet
    print("Unet successfully converted to IR")
else:
    print(f"Unet will be loaded from {UNET_OV_PATH}")


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder model to OpenVINO IR.
    Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.int64)
        # switch model to inference mode
        text_encoder.eval()
        text_encoder.config.output_hidden_states = True
        text_encoder.config.return_dict = False
        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # inputs for model tracing
                input=([1, 77],),
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
        cleanup_torchscript_cache()
        print("Text Encoder successfully converted to IR")


if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
    del pipe.text_encoder
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")

if not TEXT_ENCODER_2_OV_PATH.exists():
    convert_encoder(pipe.text_encoder_2, TEXT_ENCODER_2_OV_PATH)
    del pipe.text_encoder_2
else:
    print(f"Text encoder_2 will be loaded from {TEXT_ENCODER_2_OV_PATH}")



def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model to IR format.
    Function accepts pipeline, creates wrapper class for export only necessary for inference part,
    prepares example inputs for convert,
    Parameters:
        vae (torch.nn.Module): VAE model
        ir_path (Path): File for storing model
    Returns:
        None
    """

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latent_sample = torch.randn((1, 4, 128, 128))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(
                vae_decoder,
                example_input=latent_sample,
                input=[
                    (-1, 4, -1, -1),
                ],
            )
            
            ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE decoder successfully converted to IR")


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")


class OVStableDiffusionXLControlNetPipeline(StableDiffusionXLControlNetPipeline):
    """
    OpenVINO inference pipeline for Stable Diffusion XL with ControlNet guidence
    """

    def __init__(
        self,
        scheduler,
        unet: ov.Model,
        controlnet: ov.Model,
        tokenizer: CLIPTokenizer,   
        tokenizer_2: CLIPTokenizer,
        text_encoder: ov.Model,
        text_encoder_2: ov.Model,
        vae_decoder: ov.Model,
        device: str = "AUTO",
    ):
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.controlnet = controlnet
        self.unet = unet
        self.vae_decoder = vae_decoder
        self.scheduler = scheduler
        self.vae_scale_factor = 8
        self.vae_scaling_factor = 0.13025
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )
        self._internal_dict = {}
        self._progress_bar_config = {}

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PIL.Image.Image = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # IP adapter
        ip_adapter_scale=None,
        **kwargs,
    ):
        do_classifier_free_guidance = guidance_scale >= 1.0
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            control_guidance_start, control_guidance_end = (
                [control_guidance_start],
                [control_guidance_end],
            )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt,
            prompt_2,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            negative_prompt_2,
            lora_scale=None,
            clip_skip=clip_skip,
        )

        # 4. Prepare image
        image = self.prepare_image(
            image=image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )
        height, width = image.shape[-2:]

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            int(batch_size) * int(num_images_per_prompt),
            int(num_channels_latents),
            int(height),
            int(width),
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=generator,
            latents=latents,
        )

        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e) for s, e in zip(control_guidance_start, control_guidance_end)]
            controlnet_keep.append(keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = pooled_prompt_embeds.shape[-1]
        else:
            text_encoder_projection_dim = 1280

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds], axis=0)
            add_text_embeds = np.concatenate([negative_pooled_prompt_embeds, add_text_embeds], axis=0)
            add_time_ids = np.concatenate([negative_add_time_ids, add_time_ids], axis=0)

        add_time_ids = np.tile(add_time_ids, (batch_size * num_images_per_prompt, 1))

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                control_model_input = {
                    "sample": latent_model_input,
                    "timestep": t,
                    "encoder_hidden_states": prompt_embeds,
                    "controlnet_cond": image,
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                
                result = self.controlnet(
                    control_model_input
                    # [
                    # latent_model_input, 
                    # t, 
                    # prompt_embeds, 
                    # image,
                    # add_text_embeds, 
                    # add_time_ids,
                    # ]
                )
                down_and_mid_block_samples = [sample * controlnet_conditioning_scale for _, sample in result.items()]

                # predict the noise residual
                noise_pred = self.unet(
                    [
                        latent_model_input,
                        t,
                        prompt_embeds,
                        add_text_embeds, 
                        add_time_ids, 
                        *down_and_mid_block_samples,
                    ]
                )[0]
                
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    torch.from_numpy(noise_pred),
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False,
                )[0]
                progress_bar.update()

        if not output_type == "latent":
            image = self.vae_decoder(latents / self.vae_scaling_factor)[0]
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(torch.from_numpy(image), output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        batch_size = len(prompt)

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]

        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

        # textual inversion: procecss multi-vector tokens if necessary
        prompt_embeds_list = []
        prompts = [prompt, prompt_2]
        for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids

            prompt_embeds = text_encoder(text_input_ids)

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            hidden_states = list(prompt_embeds.values())[1:]
            if clip_skip is None:
                prompt_embeds = hidden_states[-2]
            else:
                # "2" because SDXL always indexes from the penultimate layer.
                prompt_embeds = hidden_states[-(clip_skip + 2)]

            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = np.concatenate(prompt_embeds_list, axis=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None
        if do_classifier_free_guidance and zero_out_negative_prompt:
            negative_prompt_embeds = np.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = np.zeros_like(pooled_prompt_embeds)
        elif do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(uncond_input.input_ids)
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                hidden_states = list(negative_prompt_embeds.values())[1:]
                negative_prompt_embeds = hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = np.concatenate(negative_prompt_embeds_list, axis=-1)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = np.tile(prompt_embeds, (1, num_images_per_prompt, 1))
        prompt_embeds = prompt_embeds.reshape(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = np.tile(negative_prompt_embeds, (1, num_images_per_prompt, 1))
            negative_prompt_embeds = negative_prompt_embeds.reshape(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = np.tile(pooled_prompt_embeds, (1, num_images_per_prompt)).reshape(bs_embed * num_images_per_prompt, -1)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = np.tile(negative_pooled_prompt_embeds, (1, num_images_per_prompt)).reshape(bs_embed * num_images_per_prompt, -1)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        text_encoder_projection_dim,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        return add_time_ids

def reshape(
        batch_size: int = -1,
        height: int = -1,
        width: int = -1,
        num_images_per_prompt: int = -1,
        tokenizer_max_length: int = -1,
):
    if not CONTROLNET_STATIC_OV_PATH.exists():
        controlnet = core.read_model(CONTROLNET_OV_PATH)
        def reshape_controlnet(
                model: ov.runtime.Model,
                batch_size: int = -1,
                height: int = -1,
                width: int = -1,
                num_images_per_prompt: int = -1,
                tokenizer_max_length: int = -1,
            ):
                if batch_size == -1 or num_images_per_prompt == -1:
                    batch_size = -1
                else:
                    batch_size *= num_images_per_prompt
                    # The factor of 2 comes from the guidance scale > 1
                    if "timestep_cond" not in {inputs.get_node().get_friendly_name() for inputs in model.inputs}:
                        batch_size *= 2

                height_ = height // 8 if height > 0 else height
                width_ = width // 8 if width > 0 else width
                shapes = {}
                for inputs in model.inputs:
                    shapes[inputs] = inputs.get_partial_shape()
                    if inputs.get_node().get_friendly_name() == "timestep":
                        shapes[inputs] = shapes[inputs]
                    elif inputs.get_node().get_friendly_name() == "sample":
                        shapes[inputs] = [2, 4, height_, width_]
                    elif inputs.get_node().get_friendly_name() == "controlnet_cond":
                        shapes[inputs][0] = batch_size
                        shapes[inputs][2] = height 
                        shapes[inputs][3] = width  
                    elif inputs.get_node().get_friendly_name() == "time_ids":
                        shapes[inputs] = [batch_size, 6]
                    elif inputs.get_node().get_friendly_name() == "text_embeds":
                        shapes[inputs] = [batch_size, 1280]
                    elif inputs.get_node().get_friendly_name() == "encoder_hidden_states":
                        shapes[inputs][0] = batch_size
                        shapes[inputs][1] = tokenizer_max_length
                model.reshape(shapes)
                model.validate_nodes_and_infer_types()
                
        reshape_controlnet(controlnet, batch_size, height, width, num_images_per_prompt, tokenizer_max_length)
        ov.save_model(controlnet, CONTROLNET_STATIC_OV_PATH)

    if not UNET_STATIC_OV_PATH.exists():
        unet = core.read_model(UNET_OV_PATH)
        def reshape_unet_controlnet(
            model: ov.runtime.Model,
            batch_size: int = -1,
            height: int = -1,
            width: int = -1,
            num_images_per_prompt: int = -1,
            tokenizer_max_length: int = -1,
        ):
            if batch_size == -1 or num_images_per_prompt == -1:
                batch_size = -1
            else:
                batch_size *= num_images_per_prompt
                # The factor of 2 comes from the guidance scale > 1
                if "timestep_cond" not in {inputs.get_node().get_friendly_name() for inputs in model.inputs}:
                    batch_size *= 2

            height = height // 8 if height > 0 else height
            width = width // 8 if width > 0 else width
            shapes = {}
            for inputs in model.inputs:
                shapes[inputs] = inputs.get_partial_shape()
                if inputs.get_node().get_friendly_name() == "timestep":
                    shapes[inputs] = shapes[inputs]
                elif inputs.get_node().get_friendly_name() == "sample":
                    shapes[inputs] = [2, 4, height, width]
                elif inputs.get_node().get_friendly_name() == "text_embeds":
                    shapes[inputs] = [batch_size, 1280]
                elif inputs.get_node().get_friendly_name() == "time_ids":
                    shapes[inputs] = [batch_size, 6]
                elif inputs.get_node().get_friendly_name() == "encoder_hidden_states":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][1] = tokenizer_max_length
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_1":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height 
                    shapes[inputs][3] = width    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_3":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height  
                    shapes[inputs][3] = width      
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_5":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height   
                    shapes[inputs][3] = width     
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_7":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2  
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_9":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2      
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_11":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 2 
                    shapes[inputs][3] = width // 2    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_13":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual_15":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4    
                elif inputs.get_node().get_friendly_name() == "down_block_additional_residual":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4   
                elif inputs.get_node().get_friendly_name() == "mid_block_additional_residual":
                    shapes[inputs][0] = batch_size
                    shapes[inputs][2] = height // 4 
                    shapes[inputs][3] = width // 4   

            model.reshape(shapes)
            model.validate_nodes_and_infer_types()

        reshape_unet_controlnet(unet, batch_size, height, width, num_images_per_prompt, tokenizer_max_length)
        ov.save_model(unet, UNET_STATIC_OV_PATH)

    if not TEXT_ENCODER_STATIC_OV_PATH.exists() or  not TEXT_ENCODER_STATIC_2_OV_PATH.exists():
        text_encoder = core.read_model(TEXT_ENCODER_OV_PATH)
        text_encoder_2 = core.read_model(TEXT_ENCODER_2_OV_PATH)

        def reshape_text_encoder(
            model: ov.runtime.Model, batch_size: int = -1, tokenizer_max_length: int = -1
        ):
            if batch_size != -1:
                shapes = {model.inputs[0]: [batch_size, tokenizer_max_length]}
                model.reshape(shapes)
                model.validate_nodes_and_infer_types()

        reshape_text_encoder(text_encoder, 1, tokenizer_max_length)
        reshape_text_encoder(text_encoder_2, 1, tokenizer_max_length)
        ov.save_model(text_encoder, TEXT_ENCODER_STATIC_OV_PATH)
        ov.save_model(text_encoder_2, TEXT_ENCODER_STATIC_2_OV_PATH)

    if not VAE_DECODER_STATIC_OV_PATH.exists():
        vae_decoder = core.read_model(VAE_DECODER_OV_PATH)
        def reshape_vae_decoder(model: ov.runtime.Model, height: int = -1, width: int = -1):
            height = height // 8 if height > -1 else height
            width = width // 8 if width > -1 else width
            latent_channels = 4
            shapes = {model.inputs[0]: [1, latent_channels, height, width]}
            model.reshape(shapes)
            model.validate_nodes_and_infer_types()

        reshape_vae_decoder(vae_decoder, height, width)
        ov.save_model(vae_decoder, VAE_DECODER_STATIC_OV_PATH)

reshape(
    batch_size=1,
    height=STATIC_SHAPE[0],
    width=STATIC_SHAPE[1],
    num_images_per_prompt=1,
    tokenizer_max_length=77,
    )   

def add_cache_dir(path, config):
    ov_config = config
    ov_path = path
    parent_dir = os.path.abspath(os.path.dirname(ov_path))
    ov_config["CACHE_DIR"] = os.path.join(parent_dir, "model_cache")
    return ov_config

start_time=time.time()
if NEED_STATIC:
    controlnet = core.compile_model(CONTROLNET_STATIC_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(CONTROLNET_STATIC_OV_PATH, COMPILE_CONFIG_FP16))
    unet = core.compile_model(UNET_STATIC_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(UNET_STATIC_OV_PATH, COMPILE_CONFIG_FP16))
    text_encoder = core.compile_model(TEXT_ENCODER_STATIC_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(TEXT_ENCODER_STATIC_OV_PATH, COMPILE_CONFIG_FP16))
    text_encoder_2 = core.compile_model(TEXT_ENCODER_STATIC_2_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(TEXT_ENCODER_STATIC_2_OV_PATH, COMPILE_CONFIG_FP16))
    vae_decoder = core.compile_model(VAE_DECODER_STATIC_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(VAE_DECODER_STATIC_OV_PATH, COMPILE_CONFIG_FP32))
else:
    controlnet = core.compile_model(CONTROLNET_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(COMPILE_CONFIG_FP16))
    unet = core.compile_model(UNET_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(COMPILE_CONFIG_FP16))
    text_encoder = core.compile_model(TEXT_ENCODER_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(COMPILE_CONFIG_FP16))
    text_encoder_2 = core.compile_model(TEXT_ENCODER_2_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(COMPILE_CONFIG_FP16))
    vae_decoder = core.compile_model(VAE_DECODER_OV_PATH,device_name=DEVICE_NAME, config=add_cache_dir(COMPILE_CONFIG_FP32))

tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_OV_PATH)
tokenizer_2 = CLIPTokenizer.from_pretrained(TOKENIZER_2_OV_PATH)
scheduler = EulerDiscreteScheduler.from_config(SCHEDULER_OV_PATH)


ov_pipe = OVStableDiffusionXLControlNetPipeline(
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    controlnet=controlnet,
    unet=unet,
    vae_decoder=vae_decoder,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler,
)
end_time=time.time()
print("init pipeline cost time(s): ")
print(end_time-start_time)

seed = 42
torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = 'low quality, bad quality, sketches'
controlnet_conditioning_scale = 0.5

image = load_image("./hf-logo.png")
# image = image.resize((512,512))
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
image = PIL.Image.fromarray(image)
import time
start_time=time.time()
images = ov_pipe(
    prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale
    ).images
end_time=time.time()
print("infer pipeline cost time(s): ")
print(end_time-start_time)
images[0].save(f"hug_lab.png") 