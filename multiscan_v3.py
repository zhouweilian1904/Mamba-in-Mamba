import torch
import torch.nn as nn

def to_uturn_sequence(img_tensor):
    # Assuming img_tensor is your input tensor with shape (batch, channel, height, width)
    batch, channel, height, width = img_tensor.shape

    # Permute the tensor to bring height and width to the end
    # New shape: (batch, channel, height, width) -> (batch, height, width, channel)
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    # Initialize an empty list to hold each row after applying the U-Turn pattern
    uturn_rows = []

    # Loop through each row, reversing the order in every other row
    for row_index in range(height):
        if row_index % 2 == 0:
            # Even rows remain as is
            uturn_rows.append(img_tensor[:, row_index, :, :])
        else:
            # Odd rows get reversed
            uturn_rows.append(img_tensor[:, row_index, :, :].flip(dims=[1]))

    # Stack the rows back together along the height dimension
    # And reshape to (batch, height * width, channel)
    sequence = torch.stack(uturn_rows, dim=1).reshape(batch, -1, channel)

    return sequence


# Example usage:
# img_tensor is your input tensor with shape (batch, channel, height, width)
# Replace img_tensor with your actual tensor variable
# sequence = to_uturn_sequence(img_tensor)
# Now, sequence has the shape (batch, length, dimension)


def snake_flatten(img_tensor):
    # img_tensor is expected to be of shape (batch, channel, height, width)
    batch_size, channels, height, width = img_tensor.size()

    # Permute the tensor to bring the channels to the last dimension
    img_tensor = img_tensor.permute(0, 2, 3, 1)  # New shape: (batch, height, width, channel)

    # Applying the snake pattern by reversing every other row
    for i in range(height):
        if i % 2 != 0:  # Reverse the order of pixels in every odd row
            img_tensor = img_tensor.clone()
            img_tensor[:, i] = img_tensor[:, i, :].flip(dims=[1])

    # Reshape to flatten the height and width into a single dimension, maintaining the batch and channel dimensions
    return img_tensor.reshape(batch_size, -1, channels)


def snake_unflatten(sequence, original_shape):
    # original_shape is expected to be (batch, channel, height, width)
    batch_size, channels, height, width = original_shape

    # Reshape the sequence back to (batch, height, width, channel)
    img_tensor = sequence.view(batch_size, height, width, channels)

    # Reverse the snaking pattern by flipping every alternate row back
    for i in range(height):
        if i % 2 != 0:  # Check for odd rows that were flipped
            img_tensor[:, i] = img_tensor[:, i, :].flip(dims=[1])

    # Permute back to (batch, channel, height, width)
    img_tensor = img_tensor.permute(0, 3, 1, 2)

    return img_tensor

# Example usage
# Assuming img_tensor is your input tensor
# sequence_tensor = snake_flatten(img_tensor)
# Now, sequence_tensor will be of shape (batch, height * width, channel), with the snake pattern applied.

# Define a function to create the scanning order
def create_scanning_order(image_size):
    height, width = image_size
    center_y, center_x = height // 2, width // 2

    # Scanning order starts from the edges and moves toward the center
    blue_scanning_order = [(i, j) for i in range(height) for j in range(width)]
    yellow_scanning_order = [(j, i) for j in range(width) for i in range(height)]

    # Sort by distance to the center
    blue_scanning_order.sort(key=lambda idx: abs(center_y - idx[0]) + abs(center_x - idx[1]))
    yellow_scanning_order.sort(key=lambda idx: abs(center_y - idx[0]) + abs(center_x - idx[1]))

    return blue_scanning_order + yellow_scanning_order


# Define the scan function
def scan_image(image, patch_size=1):
    batch, channel, height, width = image.size()
    scanning_order = create_scanning_order((height, width))

    patches = []
    for i, j in scanning_order:
        patch = image[:, :, i:i + patch_size, j:j + patch_size]
        patches.append(patch)

    # Stack patches into a sequence
    patches = torch.stack(patches, dim=1)  # Shape (batch, seq_len, channel, patch_size, patch_size)
    sequence = patches.view(batch, len(scanning_order), -1)  # Flatten patches into vectors
    return sequence


# Define an RNN module
class RNNModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])  # We are interested in the last time-step
        return out


image = torch.randn(100, 30, 5, 5)

# Scan the image and get the sequence of patches
patches_sequence = scan_image(image, 1)

# print('seqeunce shape', patches_sequence.shape)
