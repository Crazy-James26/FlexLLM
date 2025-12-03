from rapidstream import FloorplanConfig

config = FloorplanConfig(
    port_pre_assignments={".*": "SLOT_X0Y0:SLOT_X0Y0"},
)
config.save_to_file("floorplan_config.json")