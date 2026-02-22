import sys
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# ---------------------------------
# Path variables and input handling
# ---------------------------------
SENSOR_BLOB_PATH = '/sci/labs/sagieb/dawndude/projects/navsim_workspace/dataset/warmup_two_stage/sensor_blobs'
CHECKPOINT_PATH = "/sci/labs/sagieb/dawndude/projects/navsim_workspace/navsim/models/weights/ltf_seed_0.ckpt"

if len(sys.argv) < 2:
    print("Usage: python run_agent_on_scene.py <SCENE_PICKLE_PATH>")
    sys.exit(1)

SCENE_PICKLE_PATH = sys.argv[1]

# making sure output folder exists
output_dir = Path("temp_plots")
output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------
# Scene loading
# ---------------------------------
print("Importing Dataclasses...")
from navsim.common.dataclasses import Scene, SensorConfig
print("Done!\n#################################")

print("Pickle sanity check...")
data = pickle.load(open(SCENE_PICKLE_PATH, 'rb'))
print(data.keys())
print(data['frames'][0].keys())
print(data['frames'][0]['camera_dict'].keys())
print(data['frames'][0]['camera_dict']['cam_l0'])  # Should show the structure
print(data['frames'][0]['ego_status']['ego_pose'])
print("Length:", len(data['frames'][0]['ego_status']['ego_pose']))
print("Done!\n#################################")

# TODO For synthetic data, you can load the pickled file. For non-synthetic scenes load them directly from the sensor data folder using the from_scene_dict_list function
print("Loading Scene...")
sensor_config = SensorConfig.build_all_sensors(include=True)
sensor_config.lidar_pc = False
scene = Scene.load_from_disk(
    file_path=SCENE_PICKLE_PATH,
    sensor_blobs_path=SENSOR_BLOB_PATH,
    sensor_config=sensor_config,
)
print(scene.get_future_trajectory())
print("Done!\n#################################")

# ---------------------------------
# Scene plotting
# ---------------------------------
print("Plotting scene...")
from navsim.visualization.plots import plot_bev_frame

frame_idx = scene.scene_metadata.num_history_frames - 1 # current frame
fig, ax = plot_bev_frame(scene, frame_idx)

# Generate a filename based on the scene pickle name
scene_name = Path(SCENE_PICKLE_PATH).stem
output_file = output_dir / f"{scene_name}_bev.png"

# Save the figure
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved BEV plot to {output_file}")

# Close the figure to free memory
plt.close(fig)

print("Done!\n#################################")



# ---------------------------------
# Importing Agent and initializing
# ---------------------------------
print("Importing Transfuser Agent...")
from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
print("Done!\n#################################")

# Instantiate LFT Transfuser agent (latent=True)
print("Initializing LFT Transfuser...")
config = TransfuserConfig(latent=True)
agent = TransfuserAgent(config=config, lr=0.0, checkpoint_path=CHECKPOINT_PATH)
agent.initialize()
print("Done!\n#################################")

print("Getting agent input...")
agent_input = scene.get_agent_input()
print("Done!\n#################################")

# ---------------------------------
# Running the LFT Transfuser model on the loaded scene
# ---------------------------------
print("Running model on scene...")
if agent.requires_scene:
    trajectory = agent.compute_trajectory(agent_input, scene)
else:
    trajectory = agent.compute_trajectory(agent_input)
print("Done!\n#################################")

print(f"Trajectory output for scene {SCENE_PICKLE_PATH} (poses):")
print(trajectory.poses)



# ---------------------------------
# Plotting BEV scene with agent trajectory
# ---------------------------------

from navsim.visualization.plots import plot_bev_with_agent

fig, ax = plot_bev_with_agent(scene, agent)

# Generate a filename based on the scene pickle name
scene_name = Path(SCENE_PICKLE_PATH).stem
output_file = output_dir / f"{scene_name}_agent_bev.png"

# Save the figure
fig.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved BEV plot to {output_file}")

# Close the figure to free memory
plt.close(fig)
