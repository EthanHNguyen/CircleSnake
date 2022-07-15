from collections import OrderedDict

import torch


def rename_state_dict_keys(source, key_transformation, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)

    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.

    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"

        return old_key

    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    model = torch.load(source)
    old_state_dict = model["state_dict"]
    new_state_dict = OrderedDict()

    for key, value in old_state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    model["state_dict"] = new_state_dict

    torch.save(model, target)

def add_dla_key(old_key):
    base_key = old_key.split(".")

    if "hm" in base_key:
        base_key[0] = "ct_hm"

    if "cl" in base_key:
        base_key[0] = "radius"

    new_key = "dla." + ".".join(base_key)

    return new_key

def main():
    rename_state_dict_keys(source="../circledet_kidpath.pth", key_transformation=add_dla_key,
                           target="../circledet_kidpath_conv.pth")

if __name__ == "__main__":
    main()

