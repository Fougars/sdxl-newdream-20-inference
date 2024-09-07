import torch
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

    # Warming up the model with an empty prompt to reduce future latency
    pipeline(prompt="")

    return pipeline


def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]
