def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network.

    Args:
        layers: List of dictionaries, each describing a layer.

    Returns:
        Total number of trainable parameters (int).
    """
    params = 0
    for layer in layers:
        match layer["type"]:
            case "dense":
                i_s = layer["input_size"]
                o_s = layer["output_size"]
                b = layer.get("bias",1)
                params += (i_s * o_s) + b * o_s
            case "conv2d":
                f_in = layer["in_channels"]
                f_out = layer["out_channels"]
                b = layer.get("bias",1)
                k_h = layer["kernel_size"]
                k_w = layer["kernel_size"]
                params += (k_h * k_w * f_in * f_out) + b * f_out
    return params