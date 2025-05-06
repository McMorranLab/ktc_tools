import numpy as np
from PIL import Image

scene = np.zeros((600,1200,3))
scenex = np.arange(0,1200,1)

apHorz = 600
apVert = 300
apWidth = 10
apRad = 10
apMag = 255

scene[:,apHorz - apWidth:apHorz + apWidth,2] = apMag
scene[:,:,2] += apMag * np.exp(-((scenex-apHorz)/(apWidth * 2))**2)

scene[apVert-apRad:apVert+apRad,:,2] = 0
scene = scene.astype("uint8")

image = Image.fromarray(scene)
image.save('../../example_data/aperture.png')