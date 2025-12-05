import os
from game.game import Prsi


def main() -> None:
    env = Prsi()
    env.play()
    os.system("clear")


if __name__ == "__main__":
    main()
