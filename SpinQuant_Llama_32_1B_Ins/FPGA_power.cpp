// u280_power.cpp
#include <xrt/xrt_device.h>
#include <xrt/xrt_uuid.h>
#include <iostream>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    int dev_idx = 3;
    int interval_ms = 1000;
    if (argc > 1) dev_idx = std::stoi(argv[1]);
    if (argc > 2) interval_ms = std::stoi(argv[2]);

    // open /dev/dri/renderD* equivalent device 0
    xrt::device dev{dev_idx};

    std::cout << "Reading electrical (incl. power) for device "
              << dev_idx << " every " << interval_ms << " ms\n";

    for (;;) {
        // This returns a JSON-ish string with voltages/currents/power
        // very similar to what `xbutil examine --report electrical` shows.
        auto electrical = dev.get_info<xrt::info::device::electrical>();

        // On most Alveo cards this string includes something like:
        //  "power_watts": 123
        // just print the whole thing:
        std::cout << electrical << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
}



// g++ FPGA_power.cpp -o run/FPGA_power \
  -I/opt/xilinx/xrt/include \
  -L/opt/xilinx/xrt/lib \
  -lxrt_coreutil -lxrt_core -pthread

// ./run/FPGA_power 3 1000
