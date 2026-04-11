from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
import io
import base64

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (requires decent GPU)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16
).to("cuda")

@app.post("/make_sheer")
async def make_sheer(image: UploadFile = File(...), mask: UploadFile = File(...)):
    # Read and open images
    init_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
    mask_image = Image.open(io.BytesIO(await mask.read())).convert("RGB")
    
    # Resize images to be compatible with model (optional but recommended)
    init_image = init_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))

    prompt = "sheer fabric, translucent clothing, revealing, highly detailed anatomy, photorealistic"
    negative_prompt = "ugly, deformed, thick fabric, opaque, bad anatomy, artifacts"

    # Generate image
    result = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image=init_image, 
        mask_image=mask_image,
        strength=0.9,
        guidance_scale=7.5
    ).images[0]
    
    # Convert back to base64 to send to frontend
    buffered = io.BytesIO()
    result.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {"image": f"data:image/png;base64,{img_str}"}
