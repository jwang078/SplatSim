import zarr
import numpy as np

# --- Configuration ---
zarr_path = 'data/datasets/test_diffusion_policy_dataset.zarr'
num_episodes = 10
num_timesteps_per_episode = 50
action_dim = 8
image_height = 96
image_width = 96
image_channels = 3

# --- Dataset Generation ---
# Create the root Zarr group
# IMPORTANT! The diffusion policy repo uses zarr version 2
root_group = zarr.open(zarr_path, mode='w', zarr_version=2)

# Create the top-level 'meta' and 'data' groups
data_group = root_group.create_group('data')
meta_group = root_group.create_group('meta')

# --- Accumulate data from all episodes ---
all_keypoints = []
all_imgs = []
all_states = []
all_n_contacts = []
all_actions = []

all_episode_lengths = []

for i in range(num_episodes):
    # Simulate data for a single episode
    keypoints = np.random.rand(num_timesteps_per_episode, 3).astype(np.float32)
    imgs = np.random.randint(0, 256, size=(num_timesteps_per_episode, image_height, image_width, image_channels)).astype(np.uint8)
    states = np.random.rand(num_timesteps_per_episode, 10).astype(np.float32)
    n_contacts = np.random.randint(0, 5, size=(num_timesteps_per_episode,)).astype(np.int32)
    actions = np.random.rand(num_timesteps_per_episode, action_dim).astype(np.float32)
    episode_length = num_timesteps_per_episode

    all_keypoints.append(keypoints)
    all_imgs.append(imgs)
    all_states.append(states)
    all_n_contacts.append(n_contacts)
    all_actions.append(actions)
    all_episode_lengths.append(episode_length)

# Concatenate all episode data into single large arrays
full_keypoints = np.concatenate(all_keypoints, axis=0)
full_imgs = np.concatenate(all_imgs, axis=0)
full_states = np.concatenate(all_states, axis=0)
full_n_contacts = np.concatenate(all_n_contacts, axis=0)
full_actions = np.concatenate(all_actions, axis=0)

# Calculate the end index for each episode
episode_ends = np.cumsum(all_episode_lengths).astype(np.int64)

# --- Save the data to the Zarr file ---
# Create datasets in the 'data' group
# data_group.create_dataset(
#     'keypoints',
#     shape=full_keypoints.shape,
#     dtype=full_keypoints.dtype,
#     data=full_keypoints,
#     overwrite=True
# )
data_group.create_dataset(
    'img',
    shape=full_imgs.shape,
    dtype=full_imgs.dtype,
    data=full_imgs,
    overwrite=True
)
data_group.create_dataset(
    'state',
    shape=full_states.shape,
    dtype=full_states.dtype,
    data=full_states,
    overwrite=True
)
# data_group.create_dataset(
#     'n_contacts',
#     shape=full_n_contacts.shape,
#     dtype=full_n_contacts.dtype,
#     data=full_n_contacts,
#     overwrite=True
# )
data_group.create_dataset(
    'action',
    shape=full_actions.shape,
    dtype=full_actions.dtype,
    data=full_actions,
    overwrite=True
)

# Create the 'episode_ends' dataset in the 'meta' group
meta_group.create_dataset(
    'episode_ends',
    shape=episode_ends.shape,
    dtype=episode_ends.dtype,
    data=episode_ends,
    overwrite=True
)

print(f"Dataset successfully created at: {zarr_path}")

# --- Verification ---
read_group = zarr.open(zarr_path, mode='r')
print("\nVerifying dataset structure:")
print(f"Top-level groups: {list(read_group.keys())}")
print(f"Data-level groups/arrays: {list(read_group['data'].keys())}")
print(f"Meta-level groups/arrays: {list(read_group['meta'].keys())}")

# Check the shapes and content
# print(f"Full keypoints array shape: {read_group['data']['keypoints'].shape}")
print(f"Full image array shape: {read_group['data']['img'].shape}")
print(f"Full state array shape: {read_group['data']['state'].shape}")
# print(f"Full n_contacts array shape: {read_group['data']['n_contacts'].shape}")
print(f"Full action array shape: {read_group['data']['action'].shape}")
print(f"Episode ends array shape: {read_group['meta']['episode_ends'].shape}")
print(f"Number of episodes: {read_group['meta']['episode_ends'].shape}")