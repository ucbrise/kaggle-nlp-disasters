from typing import List

INTERVAL = 1000
HEADER = ""
buf: List[str] = []
with open("queries/logging.csv", "r") as f:
    for i, line in enumerate(f.readlines()):
        if i == 0:
            HEADER = line
        elif i % INTERVAL == 0:
            with open(f"dump/logging_{i // INTERVAL}.csv", "w") as g:
                for each in buf:
                    g.write(each)
            buf = [
                HEADER,
            ]
        buf.append(line)
