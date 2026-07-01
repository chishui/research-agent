"""Entry point: run the tool-using ResearchAgent on a topic."""

import logging
import sys

from agent import ResearchAgent


def main() -> None:
    topic = (
        " ".join(sys.argv[1:])
        or "how much protein should I take per day?"
    )
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    report = ResearchAgent().run(topic)
    print(report)


if __name__ == "__main__":
    main()
