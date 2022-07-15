import os
import unittest

import torch
from rename_state_dict_keys import rename_state_dict_keys


class TestRenameStateDictKeys(unittest.TestCase):

    def test_renaming(self):
        """
        Define some modules for the test.
        """
        import torch.nn as nn
        from torch_state_control.nn import StatefulModule


        class SimpleModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.layer = nn.Sequential(
                    nn.Linear(in_features=2, out_features=1, bias=False)
                )

            def forward(self, x):
                return self.layer(x)

        class SimpleModuleWithDropout(nn.Module):

            def __init__(self):
                super().__init__()
                self.layer = nn.Sequential(
                    nn.Dropout(p=0),
                    nn.Linear(in_features=2, out_features=1, bias=False)
                )

            def forward(self, x):
                return self.layer(x)


        """
        Try to load the state dict of SimpleModule into SimpleModuleWithDropout
        by renaming the parameters for the linear layer.
        """
        state_dict_path = './state_dict.torch'
        x = torch.tensor([1.0, 10])
        simple_module = SimpleModule()
        simple_module_with_dropout = SimpleModuleWithDropout()
        torch.save(simple_module.state_dict(), state_dict_path)

        # The test only works if at this point the results are different.
        self.assertNotEqual(simple_module(x), simple_module_with_dropout(x))

        # Before renaming, loading the state dict is expected to fail.
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            loaded_state_dict = torch.load(state_dict_path)
            simple_module_with_dropout.load_state_dict(loaded_state_dict)

        # Rename the parameters.
        def key_transformation(old_key):
            if old_key == "layer.0.weight":
                return "layer.1.weight"

            return old_key
        rename_state_dict_keys(state_dict_path, key_transformation)

        # Loading the state dict should succeed now due to the renaming.
        loaded_state_dict = torch.load(state_dict_path)
        simple_module_with_dropout.load_state_dict(loaded_state_dict)

        # Since both modules should have the same parameter values now, the
        # results should be equal.
        self.assertEqual(simple_module(x), simple_module_with_dropout(x))

        # Clean up.
        os.remove(state_dict_path)

if __name__ == '__main__':
    unittest.main()
