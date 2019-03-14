import torch

def attack(inputs, epsilon, inputs_grad):
    # Collect the element-wise sign of the data gradient
    sign_inputs_grad = inputs_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_inputs = inputs + epsilon*sign_inputs_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
    # Return the perturbed image
    return perturbed_inputs