import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../'))  # noqa

import cv2
import wave_sim2d.wave_visualizer as vis
import wave_sim2d.wave_simulation as sim
from wave_sim2d.scene_objects.source import *
from wave_sim2d.scene_objects.static_refractive_index import *

def build_scene():
    """
    This example creates the simplest possible simulation using a single emitter.
    """
    width = 512
    height = 512
    objects = [PointSource(200, 256, 0.1, 5)]
    # objects.append(StaticRefractiveIndexPolygon([[400, 255], [300, 200], [300, 300]], 1.5))
    # objects = [LineSource((200, 265), (250, 105), 0.2, 0.5)]

    return objects, width, height


def main():
    # create colormaps
    field_colormap = vis.get_colormap_lut('colormap_wave1', invert=False, black_level=-0.05)
    intensity_colormap = vis.get_colormap_lut('afmhot', invert=False, black_level=0.0)

    # build simulation scene
    scene_objects, w, h = build_scene()

    fps = 10
    out = cv2.VideoWriter('output_video0.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

    # create simulator and visualizer objects
    simulator = sim.WaveSimulator2D(w, h, scene_objects)
    visualizer = vis.WaveVisualizer(field_colormap=field_colormap, intensity_colormap=intensity_colormap)

    # run simulation
    for i in range(1000):
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

