from abc import abstractmethod
from prsi.agents.agent import Agent
from prsi.card_utils import Suit


class TrainableAgent(Agent):
    SIMPLE_HAND_INDICES = {
        Suit.BELLS: 0,
        Suit.HEARTS: 1,
        Suit.LEAVES: 2,
        Suit.ACORNS: 3,
    }

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        raise NotImplementedError("Base class cannot be trained.")

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError("Base class cannot be saved to file.")
