import os
from game.game import PrsiEnv


def main() -> None:
    env = PrsiEnv(show_ui=True)
    env.play()
    os.system("clear")


if __name__ == "__main__":
    main()
