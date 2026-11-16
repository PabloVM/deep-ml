def count_parameters(layers: list) -> int:
    """
    Count the total number of trainable parameters in a neural network.
    
    Args:
        layers: A list of dictionaries describing each layer.
                Each dict contains 'type' and layer-specific parameters.
    
    Returns:
        Total number of trainable parameters as an integer.
    """
    params = 0
    for layer in layers:

        match layer["type"]:
            case "dense":
                i_s = layer["input_size"]
                o_s = layer["output_size"]
                b = layer["use_bias"]
                params += (i_s * o_s) + b * o_s
            case "conv2d":
                f_in = layer["in_channels"]
                f_out = layer["out_channels"]
                b = layer["use_bias"]
                if type(layer["kernel_size"]) == tuple:
                    k_h,k_w = layer["kernel_size"]
                else:
                     k_h = layer["kernel_size"]
                     k_w = layer["kernel_size"]
                params += (k_h * k_w * f_in * f_out) + b * f_out
            case "embedding":
                n_emb = layer["num_embeddings"]
                emb_dim = layer["embedding_dim"]
                params += n_emb * emb_dim
    return params
