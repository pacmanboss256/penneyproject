import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def make_heatmap(data, by_tricks=True, parser=None):
    if parser is not None:
        # calculating both and only using one is faster than a branch mispredict
        print(parser.scores)
        print(parser.scores[2])
        num_tricks = sum(parser.scores[0][2]) #sum(sum(x) for _, _, x in parser.scores)
        num_cards = sum(parser.scores[0][2]) #sum(sum(x) for _, _, x in parser.scores)
    else:
        num_tricks = 0 # who knows (shrug)
        num_cards = 0
    data2 = []
    for p1_choice, p2_choice, (p1_score, p2_score, draw) in data:
        data2.append([p1_choice, p2_choice, int(p1_score), int(p2_score), int(draw)])
    data_np = np.array(data2, dtype=object)
    scores = np.array(data2, dtype=object)[:, 2:5].astype(np.int64)
    p1_win_chance = scores[:, 0] / scores.sum(axis=1)
    draw_chance   = scores[:, 2] / scores.sum(axis=1)
    trans = "".maketrans("01", "BR")

    df = pd.DataFrame({
        "Opponent Choice": [x.translate(trans) for x in data_np[:, 1]],
        "My Choice": [x.translate(trans) for x in data_np[:, 0]],
        "Score": p1_win_chance * 100,
        "Draw": draw_chance * 100,
    })

    # pivot into matrix for heatmap
    heat = df.pivot(index="Opponent Choice", columns="My Choice", values="Score")
    draw_heat = df.pivot(index="Opponent Choice", columns="My Choice", values="Draw")

    # build annotation strings like "55(12)" in the example diagram
    score_str = heat.round(0).astype("Int64").astype(str).where(heat.notna(), "")
    draw_str  = draw_heat.round(0).astype("Int64").astype(str).where(draw_heat.notna(), "")
    annot = (score_str + "(" + draw_str + ")").where(heat.notna(), "")

    plt.figure(figsize=(12, 12))

    ax = sns.heatmap(heat, annot=annot, cmap="Blues", fmt="")

    # make diagonal squares light gray
    for i in range(min(heat.shape)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="lightgray", ec="white", lw=0))
        if annot.iloc[i, i] != "":
            ax.text(i + 0.5, i + 0.5, annot.iloc[i, i],
                    ha="center", va="center")

    plt.title(f"My Chance of Win(Draw) By {'Tricks' if by_tricks else 'Cards'}\nN = {num_tricks if by_tricks else num_cards}")
    plt.xlabel("My Choice")
    plt.ylabel("Opponent Choice")
    plt.show()
