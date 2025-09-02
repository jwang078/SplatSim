import zarr
import numpy as np
import argparse
import os
import yaml
from PIL import Image, ImageOps
import pickle
from tqdm import tqdm

def main(args):
    # TODO why is this off by 1

    # traj_folder contains the image observations under traj_folder/0/images_1/base_rgb_00000.png, traj_folder/0/images_1/base_rgb_00001.png, etc
    # and contains the current joint position within traj_folder/0/00001.pkl, traj_folder/0/00002.pkl, etc
    # The number of episodes is the number of folders within traj_folder
    if args.traj_folder is None:
        with open("configs/folder_configs.yaml", 'r') as f:
            folder_configs = yaml.safe_load(f)
        traj_folder = folder_configs['traj_folder']
    else:
        traj_folder = args.traj_folder

    # --- Configuration ---
    zarr_path = f'data/datasets/{args.dataset_name}.zarr'
    print(f"Creating dataset at: {zarr_path}")

    # num_episodes is the number of folders within traj_folder
    episode_folders = sorted([name for name in os.listdir(traj_folder) if os.path.isdir(os.path.join(traj_folder, name))], key=int)
    # Default diffusion policy used 96x96
    # Splatsim images are 240x320 <-- this doesn't run on the desktop machines
    # Can also try small images with the same aspect ratios like 72x96 and 96x128
    image_height = 72
    image_width = 96

    # --- Dataset Generation ---
    # Create the root Zarr group
    # IMPORTANT! The diffusion policy repo uses zarr version 2
    root_group = zarr.open(zarr_path, mode='w', zarr_version=2)

    # Create the top-level 'meta' and 'data' groups
    data_group = root_group.create_group('data')
    meta_group = root_group.create_group('meta')


    # --- Determine total shape and create Zarr arrays ---
    total_timesteps = 0
    all_img_file_types = None
    first_image_dims = None
    # First pass: Get total number of timesteps and image types/dims
    print("Scanning folders to determine dataset size...")
    for episode_folder_base in tqdm(episode_folders):
        episode_folder = os.path.join(traj_folder, episode_folder_base)
        image_folder = os.path.join(episode_folder, 'images_1')
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        
        # Check image file types and dimensions from the first episode
        if all_img_file_types is None:
            all_img_file_types = set(['_'.join(f.split('_')[:-1]) for f in image_files])
            
            # Read first image to get dimensions
            first_image_path = os.path.join(image_folder, image_files[0])
            first_pil_img = Image.open(first_image_path)
            # Use same fitting logic as below to get correct shape
            first_pil_img = ImageOps.fit(
                first_pil_img,
                (96, 72),  # image_width, image_height, in reverse order for PIL
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5)
            )
            first_image_dims = (first_pil_img.height, first_pil_img.width, 3)

            # Find the action dimensions from the first-ish episode
            state_path = os.path.join(episode_folder, f'00001.pkl')
            with open(state_path, 'rb') as f:
                data = pickle.load(f)
            action_dim = len(data['action']) if args.action_space_type == 'end_effector_6DOF' else len(data['joint_positions'])
            state_dim = action_dim
        curr_all_img_file_types = set(['_'.join(f.split('_')[:-1]) for f in image_files])
        assert all_img_file_types == curr_all_img_file_types, f"All episodes must have the same image file types, but previous folders had {all_img_file_types} and {episode_folder_base} had types {curr_all_img_file_types}"
        num_timesteps_per_episode_per_file_type = [len([f for f in image_files if f.startswith(ftype)]) - 1 for ftype in all_img_file_types]
        assert all(n == num_timesteps_per_episode_per_file_type[0] for n in num_timesteps_per_episode_per_file_type), f"All image types must have the same number of images, but got {num_timesteps_per_episode_per_file_type} in episode {episode_folder_base}"
        num_timesteps_per_episode = num_timesteps_per_episode_per_file_type[0] - 1
        total_timesteps += num_timesteps_per_episode

    print(f"Total timesteps: {total_timesteps}")
    
    image_height, image_width, image_channels = first_image_dims
    
    # Create empty datasets with the final, known shape
    for ftype in all_img_file_types:
        data_group.create_dataset(
            ftype,
            shape=(total_timesteps, image_height, image_width, image_channels),
            dtype=np.uint8,
            chunks=(1, image_height, image_width, image_channels),  # Smaller chunks for better memory efficiency
            overwrite=True,
        )
    
    data_group.create_dataset(
        'state',
        shape=(total_timesteps, state_dim),
        dtype=np.float32,
        chunks=(1, state_dim),
        overwrite=True
    )
    data_group.create_dataset(
        'action',
        shape=(total_timesteps, action_dim),
        dtype=np.float32,
        chunks=(1, action_dim),
        overwrite=True
    )


    # --- Accumulate data from all episodes ---
    all_imgs = {}
    all_states = []
    all_actions = []

    all_episode_lengths = []

    all_img_file_types = None
    current_timestep = 0
    for episode_folder_base in tqdm(episode_folders):
        # Convert data from the traj_folder to the numpy array format
        episode_folder = os.path.join(traj_folder, episode_folder_base)
        
        # Load all images within episode_folder/images_1/base_rgb_*.png
        image_folder = os.path.join(episode_folder, 'images_1')
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
        num_timesteps_per_episode = len(image_files) - 1
        # Separate base_rgb_00000.png, base2_rgb_00000.png, wrist_rgb_00000.png, etc into different lists
        image_file_types = set(['_'.join(f.split('_')[:-1]) for f in image_files])
        if all_img_file_types is None:
            all_img_file_types = image_file_types
        else:
            assert all_img_file_types == image_file_types, f"All episodes must have the same image file types, but previous folders had {all_img_file_types} and {episode_folder_base} had types {image_file_types}"
        image_files = {
            ftype: sorted([f for f in image_files if f.startswith(ftype)]) 
            for ftype in image_file_types
        }
        num_timesteps_per_episode = [len(flist) - 1 for flist in image_files.values()]
        assert all(n == num_timesteps_per_episode[0] for n in num_timesteps_per_episode), f"All image types must have the same number of images, but got {num_timesteps_per_episode}"
        num_timesteps_per_episode = num_timesteps_per_episode[0]
        imgs_by_type = {}
        for ftype in image_files.keys():
            imgs = []
            for f in image_files[ftype][:-1]:  # Skip the last image to match the number of states/actions
                pil_img = Image.open(os.path.join(image_folder, f))
                pil_img = ImageOps.fit(
                    pil_img,
                    (image_width, image_height),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5)
                )
                img_array = np.array(pil_img)[:, :, :3]  # Keep only RGB channels

                # Add noise to the image
                img_array = img_array + args.img_noise * np.random.rand(*img_array.shape)
                imgs.append(img_array)
            imgs_by_type[ftype] = np.stack(imgs, axis=0)

        # Load all states within episode_folder/*.pkl
        all_joint_angles = []
        states = []
        actions = []
        # TODO not sure why actions start from 1 while images start from 0
        for t in range(1, num_timesteps_per_episode + 2):
            state_path = os.path.join(episode_folder, f'{t:05d}.pkl')
            with open(state_path, 'rb') as f:
                data = pickle.load(f)

            if args.action_space_type == 'end_effector_6DOF':
                # replace any "G" in the array with the float 1.0
                # TODO idk why a closed gripper produces a G string :sob:
                curr_state = np.array([1.0 if x == "G" else x for x in data['action']], dtype=np.float32)
                # curr_state = np.array(data['action'], dtype=np.float32)
            elif args.action_space_type == "joint_angles":
                curr_state = np.array(data['joint_positions'], dtype=np.float32)
            else:
                raise ValueError(f"Unknown action space type: {args.action_space_type}")
            all_joint_angles.append(curr_state)
        # TODO should it predict delta action or absolute?
        actions = np.array(all_joint_angles[1:]) # The future
        states = np.array(all_joint_angles[:-1]) # The past
        
        assert len(actions) == len(states)
        assert len(actions) == len(imgs)
        assert len(actions) == num_timesteps_per_episode

        # imgs = np.random.randint(0, 256, size=(num_timesteps_per_episode, image_height, image_width, image_channels)).astype(np.uint8)
        # states = np.random.rand(num_timesteps_per_episode, 10).astype(np.float32)
        # actions = np.random.rand(num_timesteps_per_episode, action_dim).astype(np.float32)
        episode_length = num_timesteps_per_episode

        for ftype in imgs_by_type.keys():
            if ftype not in all_imgs:
                all_imgs[ftype] = []
            all_imgs[ftype].append(imgs_by_type[ftype])
        all_states.append(states)
        all_actions.append(actions)
        all_episode_lengths.append(episode_length)

        # Write data for this episode to the Zarr file
        end_timestep = current_timestep + episode_length
        for ftype, img_data in imgs_by_type.items():
            data_group[ftype][current_timestep:end_timestep] = img_data
        
        data_group['state'][current_timestep:end_timestep] = states
        data_group['action'][current_timestep:end_timestep] = actions

        current_timestep = end_timestep

    # # Concatenate all episode data into single large arrays
    # full_imgs = {
    #     ftype: np.concatenate(all_imgs, axis=0)
    #     for ftype, all_imgs in all_imgs.items()
    # }
    # full_states = np.concatenate(all_states, axis=0)
    # full_actions = np.concatenate(all_actions, axis=0)

    # # Calculate the end index for each episode
    episode_ends = np.cumsum(all_episode_lengths).astype(np.int64)

    # # --- Save the data to the Zarr file ---
    # for ftype in full_imgs.keys():
    #     data_group.create_dataset(
    #         ftype,
    #         shape=full_imgs[ftype].shape,
    #         dtype=full_imgs[ftype].dtype,
    #         data=full_imgs[ftype],
    #         overwrite=True
    #     )
    # data_group.create_dataset(
    #     'state',
    #     shape=full_states.shape,
    #     dtype=full_states.dtype,
    #     data=full_states,
    #     overwrite=True
    # )
    # data_group.create_dataset(
    #     'action',
    #     shape=full_actions.shape,
    #     dtype=full_actions.dtype,
    #     data=full_actions,
    #     overwrite=True
    # )

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
    print(f"Full image array shape: {read_group['data']['img'].shape}")
    print(f"Full state array shape: {read_group['data']['state'].shape}")
    print(f"Full action array shape: {read_group['data']['action'].shape}")
    print(f"Episode ends array shape: {read_group['meta']['episode_ends'].shape}")
    print(f"Number of episodes: {read_group['meta']['episode_ends'].shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a diffusion policy dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to create")
    parser.add_argument("--traj_folder", type=str, default=None)
    parser.add_argument("--action_space_type", type=str, default="end_effector_6DOF", choices=["end_effector_6DOF", "joint_angles"], help="Type of action space to use")
    parser.add_argument("--img_noise", type=float, default=0.1, help="Amount of gaussian noise to add to images")
    args = parser.parse_args()
    main(args)