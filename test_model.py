from scripts.agilex_model import create_model
import torch

from get_test_inputs import get_test_frames, get_test_proprio

# Names of cameras used for visual input
CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']
config = {
    'episode_len': 1000,  # Max length of one episode
    'state_dim': 14,      # Dimension of the robot's state
    'chunk_size': 64,     # Number of actions to predict in one step
    'camera_names': CAMERA_NAMES,
    "dataset": {
        "tokenizer_max_length": 77,
        "auto_adjust_image_brightness": False,
        "image_aspect_ratio": "pad"
    }
}
pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
# Create the model with the specified configuration
model = create_model(
    args=config,
    dtype=torch.bfloat16,
    pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
    pretrained='robotics-diffusion-transformer/rdt-1b',
    control_frequency=25,
)

# Start inference process
# Load the pre-computed language embeddings
# Refer to scripts/encode_lang.py for how to encode the language instruction
lang_embeddings_path = 'embeddings/BasicMoveWhiteboardMarker.pt'
text_embedding = torch.load(lang_embeddings_path)['embeddings']

# Perform inference to predict the next `chunk_size` actions
actions = model.step(
    proprio=get_test_proprio(),
    images=get_test_frames(),
    text_embeds=text_embedding
)

print(actions)