# sheer-gen

local stable diffusion inpainting tool to swap clothes for sheer fabric.

## setup
1. `pip install -r requirements.txt`
2. run backend: `uvicorn backend.app:app --reload`
3. open `frontend/index.html` in your browser.
4. upload a pic, draw over the clothes with your mouse to create the mask, and hit generate.

requires an nvidia gpu with at least 8gb vram.
