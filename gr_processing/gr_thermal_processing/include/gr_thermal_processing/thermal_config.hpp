#ifndef THERMAL_FILTER_CONFIG_H
#define THERMAL_FILTER_CONFIG_H

namespace gr_thermal_processing{
    struct ThermalFilterConfig{
        int dilate_factor=1;
        int erosion_factor=1;
        int anchor_point=-1;
        int ddepth=-1;
        int delta=0;
        int kernel_size=3;
        int filter_iterations=10;
        int threshold=127;
        int threshold_mode=3;
        bool apply_threshold=true;
        float rescale_factor=0.5;
    };
}

#endif