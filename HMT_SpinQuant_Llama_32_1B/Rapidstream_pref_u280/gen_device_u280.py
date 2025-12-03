from rapidstream import get_u280_vitis_device_factory

# Set the Vitis platform name
factory = get_u280_vitis_device_factory("xilinx_u280_gen3x16_xdma_1_202211_1")
# Generate the virtual device in JSON format
factory.generate_virtual_device("u280_device.json")