# from torch import nn
#
#
# def get_layer_activation(activation: Optional[str] = None):
#     if activation is None:
#         return nn.Identity
#     activation = activation.lower()
#     if activation in _torch_activations_dict:
#         return getattr(nn, _torch_activations_dict[activation])
#     raise ValueError(f"Activation '{activation}' not valid.")
#
#
# class MLP(nn.Module):
#     """Simple Multi-layer Perceptron encoder with optional linear readout.
#
#     Args:
#         input_size (int): Input size.
#         hidden_size (int): Units in the hidden layers.
#         output_size (int, optional): Size of the optional readout.
#         exog_size (int, optional): Size of the optional exogenous variables.
#         n_layers (int, optional): Number of hidden layers. (default: 1)
#         activation (str, optional): Activation function. (default: `relu`)
#         dropout (float, optional): Dropout probability.
#     """
#
#     def __init__(self,
#                  input_size,
#                  hidden_size,
#                  output_size=None,
#                  exog_size=None,
#                  n_layers=1,
#                  activation='relu',
#                  dropout=0.):
#         super(MLP, self).__init__()
#
#         if exog_size is not None:
#             input_size += exog_size
#         layers = [
#             Dense(input_size=input_size if i == 0 else hidden_size,
#                   output_size=hidden_size,
#                   activation=activation,
#                   dropout=dropout) for i in range(n_layers)
#         ]
#         self.mlp = nn.Sequential(*layers)
#
#         if output_size is not None:
#             self.readout = nn.Linear(hidden_size, output_size)
#         else:
#             self.register_parameter('readout', None)
#
#     def reset_parameters(self) -> None:
#         """"""
#         for module in self.mlp._modules.values():
#             module.reset_parameters()
#         if self.readout is not None:
#             self.readout.reset_parameters()
#
#     def forward(self, x, u=None):
#         """"""
#         x = maybe_cat_exog(x, u)
#         out = self.mlp(x)
#         if self.readout is not None:
#             return self.readout(out)
#         return out