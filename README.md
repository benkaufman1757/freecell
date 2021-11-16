# freecell
Python Freecell Solver

Very early version right now. You can pick a board by changing the file path in freecell.py

If you want to play a game of freecell you can, the controls are in the code right now -- I mostly just used the play() function to debug stuff.

Right now board 1 takes 10 seconds on my computer, board 2 takes a while but is supposedly a hard board. Copying the objects takes the most time out of anything.

The heuristic function I've called score() was just something I made up and I think it works ok, I'm thinking of doing some "parameter sweep" or something to see if I can find better values.

Also, I need a way to track/store the solution and then display all the moves in the solution.
