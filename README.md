# freecell
Python Freecell Solver

Early version right now, it works ok. I've added some command line options you can check by running:

`python3 freecell.py -h
`

If you want to play a game of freecell you can, the controls are in the code right now.

Boards are solved more quickly, board 1 takes about 2 seconds compared to 10 seconds. Copying the objects takes the most time out of anything.

The heuristic function I've called score() was just something I made up and I think it works ok, I'm thinking of doing some "parameter sweep" or something to see if I can find better values.
I've now split up the score function into a few different types
* simple: 52 - number of cards in all foundations
* simple+ (admissable): (52 - number of cards in all foundations) + number of moves
* bennaive: in descending order of importance, good if there are cards in the foundation, good if there are free spaces available, good if there are kings are the first card in a column
