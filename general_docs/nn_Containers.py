
# * torch.nn.Module [https://pytorch.org/docs/stable/nn.html]

# * Module
# add_module(name, module)
# adds a child module to the current module
#
# apply(fn)
# applies function fn to all child modules (returned by children)
#
# buffers() / named_buffers()
# iterator over all buffers (for example the mean values for BatchNorm layers)
#
# children() / named_children()
# iterator over all children modules
#
# cpu/cuda(device=None)
# moves module to cpu/cuda. if cuda device (int) != None, moves to that cuda device
#
# double()
# cast all parameters to double
#
# eval()
# sets module to evaluation mode (affects only some layers, e.g., batchnorm, dropout)
#
# extra_repr()
# better __repr__ (needs to manually implement)
#
# float()
# cast all parameters to float
#
# forward(*input)
# defines the forward operations
#
# half()
# cast parameters to half float
#
# load_state_dict(state_dict, strict=True)
# copies parameters and bufferes from state_dict into module
# if strict is true, then the keys of state_dict
# must exactly match the keys returned by this module’s state_dict() function.
#
# named_parameters() / parameters()
# iterator over parameters
#
# register_forward_hook(hook)
# register a function that will be called everytime after forward()
# hook = (module, input, output) -> None
#
# register_forward_pre_hook()
# register a function that will be called before forward()
# hook = (module, input) -> None
#
# register_parameter(name, param)
# register a parameter, I can't imagine any use for this...
#
# state_dict()
# Returns a dictionary containing a whole state of the module.
#
# to()
# moves and/or casts parameters and buffers
# examples: to(device), to(dtype), to(device=device, dtype=dtype, non_blocking=False)
# When non_blocking is set, it tries to convert/move asynchronously
# with respect to the host if possible, e.g., moving CPU Tensors with
# pinned memory to CUDA devices.
#
# train()
# sets the model to training mode. Affects some layers, such as, Dropout and BatchNorm
#
# type()
# casts all paramaters and buffers to some type
#
# zero_grad()
# Sets gradients of all model parameters to zero


# * Sequential
# A sequential container. Modules will be added to it in the order
# they are passed in the constructor.
# Alternatively, an ordered dict of modules can also be passed in.
# * Note that since Sequential is a subclass of Module,
# * we can call add_module to add more layers
# * to an existing Sequential object

# # Example of using Sequential
# model = nn.Sequential(
#           nn.Conv2d(1,20,5),
#           nn.ReLU(),
#           nn.Conv2d(20,64,5),
#           nn.ReLU()
#         )

# # Example of using Sequential with OrderedDict
# model = nn.Sequential(OrderedDict([
#           ('conv1', nn.Conv2d(1,20,5)),
#           ('relu1', nn.ReLU()),
#           ('conv2', nn.Conv2d(20,64,5)),
#           ('relu2', nn.ReLU())
#         ]))

# *ModuleList
# Holds submodules in a list.
# append/extend/insert

# *ModuleDict
# Holds submodules in a dict.
# clear/items/keys/pop/update/values
# useful example
# # Example
# MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.choices = nn.ModuleDict({
#                 'conv': nn.Conv2d(10, 10, 3),
#                 'pool': nn.MaxPool2d(3)
#         })
#         self.activations = nn.ModuleDict([
#                 ['lrelu', nn.LeakyReLU()],
#                 ['prelu', nn.PReLU()]
#         ])

#     def forward(self, x, choice, act):
#         x = self.choices[choice](x)
#         x = self.activations[act](x)
#         return x

# *ParameterList / ParameterDict
