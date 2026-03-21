# penneyproject

Penney's Game is a non-transitive game, where the player who goes second always has a better option than the player who chooses first. The original game uses a coin, and each player picks a sequence of 3 coin flips, Heads or Tails. In the Humble-Nishiyama version, decks of cards are used instead, where players choose a sequence of black or red cards. Cards are drawn from the top of the deck until a sequence of 3 matches a player's choice. The player whose sequence was found wins that trick, and the process repeats until the deck ends. The other variation is nearly identical, except instead of winning one point per trick, you win the number of cards drawn since the last sequence, up to and including the three in the match. This leads to fewer ties, and a more interesting result for optimal choice.

To get this project working, you can run these commands:
```bash
uv build && uv install && uv run main.py
```

Alternatively, if above doesn't work, then to compile the parser, assuming your cwd is this root, do this command, restart your notebook, and it should work from there:
```bash
cd src && python3 setup.py build_ext --inplace && cd ..
```

Our trick-based results agree with the published H-N game. They show the same structure and the same advantage for the second player. The optimal second-player response  for the trick-based game follows the rule that if player 1 chooses x1, x2, x3, then player 2 should choose opposite(x2), x1, x2, meaning you flip the middle symbol of player 1's sequence, put that flipped symbol first, and then copy player 1's first two symbols. Our heatmap confirms that this rule gives the optimal response in every case for the original trick-scored game. The card-scored version is very similar overall and still strongly favors the second player, but it is not identical: the same rule remains optimal in most cases, while our results show exceptions for BRB and RBR, and the second-player edge is generally even larger than in the trick-based version. Because of the exceptions, we can formulate a new rule to cover all the cases in the card-based scoring system. First, let M = majority(x1, x2, x3). Then the optimal response follows the rule that player 2 should choose opposite(M), majority(x1, x2, opposite(x3)), M, meaning you take the majority color in player 1's sequence, put its opposite first, then take the majority color after flipping the third symbol and put that second, and finally put the original majority color third.
