# This code uses the SLEAP AI library for multi-animal pose tracking.
# Please cite the following work when using SLEAP:
# Talmo D. Pereira et al., "SLEAP: A deep learning system for multi-animal pose tracking," Nature Methods, 2022. 
# DOI: https://doi.org/10.1038/s41592-021-01210-2

import h5py
import sleap
import os

# this is meant to run sleap and output the necessary data for the other program to generate graphs/stats about it

def analyze_video(video_path):
    """
    Analyze a video of larvae using SLEAP AI and save the (x, y) larvae positions across frames.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the output data.
    """
    output_dir = os.path.join(os.getcwd(), "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_path = '/Users/delia/Desktop/UVA/CONDRON LAB/condron_lab_sleap_analysis/Model/labels.v001.slp'
    model = sleap.load_model(model_path)
    
    # Run inference on the video
    print(f"Processing video: {video_path}")
    predictions = model.predict(video_path, batch_size=4)
    
    # Save predictions as an HDF5 file
    output_h5 = os.path.join(output_dir, "larvae_positions.h5")
    predictions.save(output_h5)
    print(f"Predictions saved to: {output_h5}")
    
    # Optionally, extract positions and print or process them
    with h5py.File(output_h5, "rb") as f:
        tracks = f["tracks"][:]  # Shape: (frames, nodes, 2, instances)
        num_frames, num_nodes, _, num_instances = tracks.shape
        
        print(f"Number of frames: {num_frames}")
        print(f"Number of larvae (instances): {num_instances}")
        
        # Example: Extract (x, y) positions for the first instance in each frame
        positions = []
        for frame in range(num_frames):
            frame_positions = []
            for instance in range(num_instances):
                positions = []
                for frame in range(num_frames):
                    frame_positions = []
                    for instance in range(num_instances):
                        # Extract (x, y) coordinates for both nodes
                        spiracle_x, spiracle_y = tracks[frame, 0, :, instance]  # Node 0 (spiracle)
                        mouthhook_x, mouthhook_y = tracks[frame, 1, :, instance]  # Node 1 (mouth hook)
                        
                        frame_positions.append({
                            "spiracle": (spiracle_x, spiracle_y),
                            "mouth_hook": (mouthhook_x, mouthhook_y)
                        })
                    positions.append(frame_positions)

        # Save positions to a text file
        output_txt = os.path.join(output_dir, "positions.txt")
        with open(output_txt, "w") as file:
            for frame_idx, frame_positions in enumerate(positions):
                file.write(f"Frame {frame_idx}:\n")
                for instance_idx, pos in enumerate(frame_positions):
                    file.write(f"  Instance {instance_idx}: Spiracle x={pos['spiracle'][0]}, y={pos['spiracle'][1]} | "
                            f"Mouth Hook x={pos['mouth_hook'][0]}, y={pos['mouth_hook'][1]}\n")
        print(f"Positions saved to: {output_txt}")
        return output_h5

# Example:
video_path = '/Users/delia/Desktop/UVA/CONDRON LAB/condron_lab_sleap_analysis/Model/training-movies/s8-movie copy.mp4'
analyze_video(video_path)
