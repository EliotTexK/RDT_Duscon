# Not meant to test if inference gives useful outputs
# Just meant to test if the model can do inference without raising an exception

import PIL
import torch
from typing import List

def get_test_frames() -> List[PIL.Image]:
    # Load and immediately evaluate the first image
    img1 = PIL.Image.open('random_image_1.png')
    img1.load()  # Force immediate evaluation

    # Load and immediately evaluate the second image
    img2 = PIL.Image.open('random_image_2.png')
    img2.load()  # Force immediate evaluation

def get_test_proprio(): # What type is expected here?!?!
    # Create an example proprio tensor with realistic joint angles
    # Units are typically in radians for revolute joints
    proprio = torch.tensor([
        # Left arm joints (6 joints, indices 0-5)
        0.0,      # left_arm_joint_0_pos - base rotation
        -0.5,     # left_arm_joint_1_pos - shoulder lift
        1.2,      # left_arm_joint_2_pos - elbow
        0.3,      # left_arm_joint_3_pos - wrist rotation 1
        -0.2,     # left_arm_joint_4_pos - wrist rotation 2
        0.1,      # left_arm_joint_5_pos - wrist rotation 3
        
        # Left gripper (1 value, index 6)
        2.5,      # left_gripper_open - gripper state (0=closed, ~4.79=fully open)
        
        # Right arm joints (6 joints, indices 7-12)
        0.2,      # right_arm_joint_0_pos - base rotation
        -0.6,     # right_arm_joint_1_pos - shoulder lift
        1.0,      # right_arm_joint_2_pos - elbow
        0.4,      # right_arm_joint_3_pos - wrist rotation 1
        -0.3,     # right_arm_joint_4_pos - wrist rotation 2
        0.0,      # right_arm_joint_5_pos - wrist rotation 3
        
        # Right gripper (1 value, index 13)
        3.0,      # right_gripper_open - gripper state (0=closed, ~4.79=fully open)
    ], dtype=torch.float32)