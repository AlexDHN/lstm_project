kwargs = dict(
    learning_rate = 0.0075,
    batch_size = 1000,
    num_epochs = 100,
    window_size = 30,
    hidden_number = 1,
    hidden_size = 3,
    weight_decay = 1e-5,
    override = False,
    path = "models/"
)

init_functions = [
    "random_normal",
    "random_uniform",
    "truncated_normal",
    "zeros",
    "ones",
    "glorot_normal",
    "glorot_uniform",
    "identity",
    "orthogonal",
    "constant_",
    "variance_scaling",
]