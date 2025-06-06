import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../'))

import numpy as np
import cv2

import wave_sim2d.wave_visualizer as vis
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.static_image_scene import StaticImageScene


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


def main():
    # Set scene image path. The image encodes refractive index, dampening and emitters in its color channels
    # see 'static_image_scene.StaticImageScene' class for a more detailed description.
    # please take a look at the image to understand what is happening in the simulation
    scene_image_path = '../../example_data/scene_lens_doubleslit.png'

    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # reset random number generator
    np.random.seed(0)

    # build simulation scene
    scene_objects, w, h = build_scene(scene_image_path)

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    fps = 30
    out = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
    
    # run simulation
    for i in range(2000):
        simulator.update_scene()
        simulator.update_field()
        visualizer.update(simulator)

        # show field
        frame_field = visualizer.render_field(1.0)
        # cv2.imshow("Wave Simulation Field", frame_field)
        out.write(frame_field)


        # show intensity
        # frame_int = visualizer.render_intensity(1.0)
        # cv2.imshow("Wave Simulation Intensity", frame_int)

        cv2.waitKey(1)

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

