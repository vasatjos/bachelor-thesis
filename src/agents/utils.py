import collections
import torch


Transition = collections.namedtuple(
    "Transition",
    ["state", "action_idx", "reward", "done", "next_state", "next_valid_actions"],
)


class Network(torch.nn.Module):
    def __init__(self, hidden_size: int, hidden_count: int, output_size: int) -> None:
        if hidden_count < 1 or hidden_size < 1:
            raise ValueError("Invalid network parameters!")

        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_size), torch.nn.ReLU()
        )

        for _ in range(hidden_count):
            self.net.append(torch.nn.Linear(hidden_size, hidden_size))
            self.net.append(torch.nn.ReLU())

        self.net.append(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
