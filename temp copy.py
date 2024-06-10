import torch

# Example tensors
infostate_matrix = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
                                 [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])
permutation = torch.tensor([[2, 1, 0], [0, 2, 1]])

# Use torch.take_along_dim to permute infostate_matrix on the last dimension
permuted_infostate_matrix = torch.take_along_dim(infostate_matrix, permutation.unsqueeze(-1), dim=-1)

print(permuted_infostate_matrix)
print(permuted_infostate_matrix.shape)