import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../')) # noqa

import cv2
import numpy as np
import cupy as cp
import wave_sim2d.wave_visualizer as vis
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.source import *
from wave_sim2d.scene_objects.static_refractive_index import *
from wave_sim2d.scene_objects.static_dampening import *
from wave_sim2d.scene_objects.static_image_scene import *

def build_scene(scene_image_path):
    """
    This example uses the 'old' image scene description. See 'StaticImageScene' for more information.
    """
    # load scene image
    scene_image = cv2.cvtColor(cv2.imread(scene_image_path), cv2.COLOR_BGR2RGB)

    # create the scene object list with an 'StaticImageScene' entry as the only scene object
    # more scene objects can be added to the list to build more complex scenes
    scene_objects = [StaticImageScene(scene_image, source_fequency_scale=2.0)]

    return scene_objects, scene_image.shape[1], scene_image.shape[0]


def build_scene():
    """
    diffuse incoherent point source and aperture
    """
    scene_image_path = '../../example_data/keplerianTelescope.png'
    width = 1920
    height = 1080

    objects = []

    # Add a static dampening field without any dampending in the interior (value 1.0 means no dampening)
    # However a dampening layer at the border is added to avoid reflections (see parameter 'border thickness')
    objects.append(StaticDampening(np.ones((height, width)), 32))
    # add a constant refractive index field
    objects.append(StaticRefractiveIndex(np.full((height, width), 1.0)))

    #need to put any dampening objects after the generalized static dampening object
    scene_image = cv2.cvtColor(cv2.imread(scene_image_path), cv2.COLOR_BGR2RGB)
    scene_image = cv2.resize(scene_image, (width, height))
    objects.append(StaticImageScene(scene_image, source_fequency_scale=2.0))

    #add a line source since my green pixels didnt do what I want
    objects.append(LineSource((100, (height/2) - 400), (100, (height/2) + 400), .4, amplitude=0.3))

    return objects, width, height

def show_field(field, brightness_scale):
    gray = (cp.clip(field*brightness_scale, -1.0, 1.0) * 127 + 127).astype(np.uint8)
    img = gray.get()
    cv2.imshow("Strain Simulation Field", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    write_videos = False
    write_video_frame_every = 2

    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave2', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # build simulation scene
    scene_objects, w, h = build_scene()

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    simulator.dt = .5
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    fps = 60
    videoName = 'keplerianTelescope.mp4'
    out = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

    # run simulation
    for i in range(10000):
        simulator.update_scene()
        simulator.update_field()

        visualizer.update(simulator)
        # # show field
        # frame_field = visualizer.render_field(1.0)
        # cv2.imshow("Wave Simulation Field", frame_field)
        # show field
        frame_field = visualizer.render_field(1.0)

        
        out.write(frame_field)


        # # show intensity
        # frame_int = visualizer.render_intensity(1.0)
        # cv2.imshow("Wave Simulation Intensity", frame_int)


        cv2.waitKey(1)

    out.release()
    cv2.destroyAllWindows()

main()

