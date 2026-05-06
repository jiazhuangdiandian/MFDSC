import torch
import torch.nn as nn

class Reject(nn.Module):
    def __init__(self, c, num_channels_to_remove):
        super(Reject, self).__init__()
        self.aag = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fcn = nn.Sequential(
            nn.Linear(c, 2 * c),
            nn.ReLU(),
            nn.Linear(2 * c, c))
        self.num_channels_to_remove = num_channels_to_remove

    def forward(self, z):
        # Adaptive average pooling
        out = self.aag(z)
        print("Adaptive average pooling output shape:", out.shape)
        out = torch.squeeze(out, dim=2).squeeze(dim=2)  # Remove dimensions of size 1

        out_fcn = self.fcn(out)
        print("Fully connected output shape:", out_fcn.shape)

        # Get the indices of the smallest values in out_fcn
        _, indices = torch.topk(torch.abs(out_fcn), self.num_channels_to_remove, largest=False, dim=1)
        print("Indices shape:", indices.shape)

        # Create a mask to keep only the channels not in the indices
        mask = torch.ones(out_fcn.size(), dtype=torch.bool, device=out_fcn.device)
        mask.scatter_(1, indices, False)
        print("Mask shape after setting indices to False:", mask.shape)

        # Apply mask to the input tensor z to keep or remove channels
        z_filtered = torch.masked_select(z, mask.unsqueeze(2).unsqueeze(3)).view(z.size(0), -1, z.size(2), z.size(3))
        print("Filtered z shape:", z_filtered.shape)

        return z_filtered

# Example usage
z = torch.randn(10, 16, 8, 8)  # Example input tensor
model = Reject(c=16, num_channels_to_remove=4)
output = model(z)
print("Output shape:", output.shape)  # Output shape should be (10, 12, 8, 8)
