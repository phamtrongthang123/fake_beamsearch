import torch 
import einops
a = torch.tensor([1,2,3,4,5,6,7,8,9,10]).view(2,5)  
print(einops.repeat(a, 'b n -> (b r) n ', r=2))
print(torch.cat([a,a], dim=0))
# print(a.expand(4,-1)) # error, pytorch only allow expand if num  = 1, not 2,3 ... 
print(a.repeat(2,1))
print(a.repeat_interleave(2, dim=0))
# tensor([[ 1,  2,  3,  4,  5],
#         [ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10],
#         [ 6,  7,  8,  9, 10]])
# tensor([[ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10],
#         [ 1,  2,  3,  4,  5],
#         [ 6,  7,  8,  9, 10]])