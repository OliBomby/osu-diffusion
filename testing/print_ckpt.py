import torch


def print_state_dict(obj):
    for param_tensor in obj:
        param = obj[param_tensor] if isinstance(obj, dict) else param_tensor
        if isinstance(param, dict) or isinstance(param, list):
            print(param_tensor, "\t", "state dict:")
            print_state_dict(param)
        elif isinstance(param, torch.Tensor):
            print(param_tensor, "\t", param.size())
        else:
            print(param_tensor, "\t", param)


ckpt = torch.load("D:\\DiT-B-0130000.pt")
embedding_table = ckpt["ema"]["y_embedder.embedding_table.weight"]

# Print model's state_dict
print("Model's state_dict:")
print_state_dict(ckpt)
