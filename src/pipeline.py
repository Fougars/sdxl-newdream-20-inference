import torch
from typing import List
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pipelines.models import TextToImageRequest
from torch import Generator

def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "./models/newdream-sdxl-20",
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")

    # Optimization: Enable torch.compile for the unet
    try:
        pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
    except Exception as e:
        print(f"Warning: Could not compile unet: {e}")

    # Optimization: Enable memory efficient attention
    try:
        if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
            pipeline.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(f"Warning: Could not enable xformers memory efficient attention: {e}")

    # Optimization: Enable CPU offload if necessary
    if torch.cuda.mem_get_info()[0] / 1024**3 < 14:  # Check if less than 14GB free
        try:
            pipeline.enable_model_cpu_offload()
        except Exception as e:
            print(f"Warning: Could not enable model CPU offload: {e}")

    # Optimization: Enable attention slicing
    pipeline.enable_attention_slicing()

    # Optimization: Enable VAE slicing
    pipeline.enable_vae_slicing()

    # Optimization: Use CUDA graphs for faster inference
    try:
        pipeline.enable_cuda_graph()
    except Exception as e:
        print(f"Warning: Could not enable CUDA graph: {e}")

    # Warming up the model with an empty prompt to reduce future latency
    pipeline(prompt="")

    return pipeline

# Optimization: Implement caching mechanism
prompt_cache = {}

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    cache_key = (request.prompt, request.negative_prompt, request.width, request.height, request.seed)
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]

    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    with torch.inference_mode(), torch.cuda.amp.autocast():
        image = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            generator=generator,
            num_inference_steps=30,  # Reduced for faster inference
        ).images[0]

    prompt_cache[cache_key] = image
    return image

# Optimization: Implement batch inference
def infer_batch(requests: List[TextToImageRequest], pipeline: StableDiffusionXLPipeline) -> List[Image]:
    prompts = [r.prompt for r in requests]
    negative_prompts = [r.negative_prompt for r in requests]
    widths = [r.width for r in requests]
    heights = [r.height for r in requests]
    seeds = [r.seed for r in requests]

    generators = [Generator(pipeline.device).manual_seed(seed) if seed else None for seed in seeds]

    with torch.inference_mode(), torch.cuda.amp.autocast():
        images = pipeline(
            prompt=prompts,
            negative_prompt=negative_prompts,
            width=widths,
            height=heights,
            generator=generators,
            num_inference_steps=30,  # Reduced for faster inference
        ).images

    return images
