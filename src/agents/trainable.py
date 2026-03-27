from abc import abstractmethod
from prsi.agents.agent import Agent


class TrainableAgent(Agent):
    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be trained.")

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")
