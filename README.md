pip install DeOLDify

!git clone https://github.com/jantic/DeOldify.git DeOldify 

cd DeOldify


from deoldify import device
from deoldify.device_id import DeviceId

device.set(device=DeviceId.GPU0)

import torch

if not torch.cuda.is_available():
    print('GPU not available.')


!pip install -r requirements-colab.txt

import fastai
from deoldify.visualize import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")


!mkdir 'models'
!wget https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -O ./models/ColorizeArtistic_gen.pth

colorizer = get_image_colorizer(artistic=True)

source_url = 'https://picjumbo.com/wp-content/uploads/lion-stare-serious-portrait-dark-black-and-white-free-photo.jpg' #@param {type:"string"}
render_factor = 22  #@param {type: "slider", min: 7, max: 40}
watermarked = True #@param {type:"boolean"}


if source_url is not None and source_url !='':
    image_path = colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
    show_image_in_notebook(image_path)
else:
    print('Provide an image url and try again.')

 
for i in range(10,40,2):
    colorizer.plot_transformed_image('test_images/image.png', render_factor=i, display_render_factor=True, figsize=(8,8))
