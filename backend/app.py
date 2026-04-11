from fastapi import fastapi, uploadfile, file
from fastapi.middleware.cors import corsmiddleware
from diffusers import stablediffusioninpaintpipeline
import torch
from pil import image
import io
import base64

app = fastapi()

# allow frontend to talk to backend
app.add_middleware(
    corsmiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load the model (requires decent gpu)
pipe = stablediffusioninpaintpipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16
).to("cuda")

@app.post("/make_sheer")
async def make_sheer(image: uploadfile = file(...), mask: uploadfile = file(...)):
    init_image = image.open(io.bytesio(await image.read())).convert("rgb")
    mask_image = image.open(io.bytesio(await mask.read())).convert("rgb")
    
    prompt = "sheer fabric, translucent clothing, revealing, highly detailed anatomy, photorealistic"
    negative_prompt = "ugly, deformed, thick fabric, opaque, bad anatomy, artifacts"

    result = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        image=init_image, 
        mask_image=mask_image
    ).images[0]
    
    # convert back to base64 to send to frontend
    buffered = io.bytesio()
    result.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return 
