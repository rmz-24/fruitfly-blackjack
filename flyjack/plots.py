"""Figures for the README (written to ``figures/``).

    python -m flyjack.plots
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .anatomy import ROOT
from .blackjack import expected_return
from .record import load_responses
from .strategies import BASELINES, basic_strategy
from .train import RESULTS

FIGURES = ROOT / "figures"

# Reference data-viz palette (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
STAND_C, MID_C, HIT_C = "#256abf", "#f0efec", "#e34948"
HIT_MAP = LinearSegmentedColormap.from_list("stand_hit", [STAND_C, MID_C, HIT_C])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def learning_curves():
    log = json.load(open(RESULTS / "training.json"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    hands = np.array([h["hands"] for h in log["runs"][0]["history"]])
    ev = np.array([[h["expected_return"] for h in r["history"]] for r in log["runs"]])
    ax.fill_between(hands, ev.min(0), ev.max(0), color=SERIES[0], alpha=0.18, lw=0)
    ax.plot(hands, ev.mean(0), color=SERIES[0], lw=2, label=f"fly (mean of {len(ev)} seeds)")
    refs = [("basic strategy (optimal)", INK, "-"), ("dealer rule (hit < 17)", SERIES[1], "--"),
            ("never bust", SERIES[2], "--"), ("random", SERIES[4], ":")]
    for name, color, ls in refs:
        v = expected_return(BASELINES[name]() if name != "random" else (lambda o: 0.5))
        ax.axhline(v, color=color, lw=1.2, ls=ls)
        ax.text(hands[-1], v, f" {name} {v:+.3f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_xlim(0, hands[-1])
    ax.set_ylim(-0.4, 0.0)
    ax.set_xlabel("hands played")
    ax.set_ylabel("expected reward per hand")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.set_title("The fly learns Blackjack through dopamine-gated plasticity", loc="left")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=150)
    plt.close(fig)


def _grid(p, usable):
    sums = range(12, 22) if usable else range(4, 22)
    return sums, np.array([[p[(s, d, usable)] for d in range(1, 11)] for s in sums])


def policy_heatmaps():
    ev = json.load(open(RESULTS / "evaluation.json"))
    fly = {(s, d, bool(u)): p for s, d, u, p in ev["p_hit"]}
    basic = {o: float(basic_strategy(o)) for o in fly}
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 9), gridspec_kw={"height_ratios": [18, 10]})
    for col, (title, pol) in enumerate([("The fly", fly), ("Basic strategy", basic)]):
        for row, usable in enumerate([False, True]):
            ax = axes[row, col]
            sums, g = _grid(pol, usable)
            im = ax.imshow(g, cmap=HIT_MAP, vmin=0, vmax=1, origin="lower", aspect="auto",
                           extent=(0.5, 10.5, sums[0] - 0.5, sums[-1] + 0.5))
            ax.grid(False)
            ax.set_xticks(range(1, 11), ["A"] + [str(d) for d in range(2, 11)])
            ax.set_yticks(list(sums))
            ax.tick_params(length=0, labelsize=8)
            if col == 0:
                _, b = _grid(basic, usable)
                for i, s in enumerate(sums):
                    for j in range(10):
                        if (g[i, j] > 0.5) != (b[i, j] > 0.5):
                            ax.add_patch(plt.Rectangle((j + 0.5, s - 0.5), 1, 1, fill=False,
                                                       ec=INK, lw=1.6))
            ax.set_title(f"{title}: {'soft' if usable else 'hard'} hands", loc="left", fontsize=10)
            ax.set_xlabel("dealer upcard")
            if col == 0:
                ax.set_ylabel("player sum")
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.03, pad=0.07,
                      ticks=[0, 0.5, 1])
    cb.ax.set_xticklabels(["stand", "50/50", "hit"])
    cb.outline.set_visible(False)
    fig.suptitle("Learned strategy vs. basic strategy", x=0.06, y=0.975, ha="left",
                 fontweight="bold")
    fig.text(0.06, 0.945,"Fly: probability of hitting over held-out neural trials. "
             "Outlined cells: the fly's majority choice differs from basic strategy.",
             fontsize=8.5, color=INK2)
    fig.savefig(FIGURES / "policy_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def kenyon_code():
    obs, d = load_responses()
    kc = d["kc"].astype(np.float32)
    # order: hard sums then soft, dealer inside -> block structure by player sum
    order = sorted(range(len(obs)), key=lambda i: (obs[i][2], obs[i][0], obs[i][1]))
    m = kc.mean(1)[order]
    corr = np.corrcoef(m)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})
    im = a1.imshow(corr, cmap=LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#6da7ec", "#0d366b"]),
                   vmin=0, vmax=1)
    a1.grid(False)
    n_hard = sum(1 for o in obs if not o[2])
    a1.axhline(n_hard - 0.5, color=SERIES[1], lw=1)
    a1.axvline(n_hard - 0.5, color=SERIES[1], lw=1)
    a1.set_xticks([n_hard / 2, n_hard + (len(obs) - n_hard) / 2], ["hard hands", "soft hands"])
    a1.set_yticks([n_hard / 2, n_hard + (len(obs) - n_hard) / 2], ["hard", "soft"], rotation=90, va="center")
    a1.tick_params(length=0)
    a1.set_title("KC population similarity between card states", loc="left")
    fig.colorbar(im, ax=a1, fraction=0.046, label="correlation").outline.set_visible(False)

    active = (kc > 0).mean(2).ravel() * 100
    a2.hist(active, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    a2.axvline(active.mean(), color=INK, lw=1.2, ls="--")
    a2.text(active.mean(), a2.get_ylim()[1] * 0.95, f"  mean {active.mean():.1f}%", fontsize=9, va="top")
    a2.set_xlabel("% of the 5,177 Kenyon cells that spike")
    a2.set_ylabel("stimulus presentations")
    a2.set_title("Sparse KC coding (280 states x 32 trials)", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "kenyon_code.png", dpi=150)
    plt.close(fig)


CARD = {1: "A", 10: "10"}
POOLS_LABEL = {"hit": "2 x MBON01", "stand": "2 x MBON03"}


def _cards(cards):
    return " ".join(CARD.get(c, str(c)) for c in cards)


def hand_raster():
    """The spiking brain playing one hand: spikes of the stimulated projection
    neurons, Kenyon cells, the HIT/STAND MBON pools and descending neurons."""
    path = RESULTS / "closed_loop_hand.npz"
    if not path.exists():
        return
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    steps, groups = meta["steps"], {k: np.array(v) for k, v in meta["groups"].items()}
    rasters = [d[f"raster{i}"] for i in range(len(steps))]
    gap = 60.0
    order = [("dn", "descending neurons", SERIES[2]), ("stand", "STAND MBONs", STAND_C),
             ("hit", "HIT MBONs", HIT_C), ("kc", "Kenyon cells", SERIES[3]),
             ("pn", "projection neurons", SERIES[4])]
    # rows: only neurons that spiked at least once, pools always shown
    rows, y0, ticks = {}, 0, []
    heights = {"dn": 1, "stand": 22, "hit": 22, "kc": 1, "pn": 1}
    for key, label, _ in order:
        ids = groups[key]
        fired = np.unique(np.concatenate([r[np.isin(r[:, 0], ids), 0] for r in rasters]))
        show = ids if key in ("hit", "stand") else fired
        h = heights[key]
        rows[key] = {int(n): y0 + i * h + h / 2 for i, n in enumerate(show)}
        note = f" ({POOLS_LABEL[key]})" if key in POOLS_LABEL else f"\n({len(fired)} of {len(ids)} spiked)"
        ticks.append((y0 + len(show) * h / 2, label + note))
        y0 += len(show) * h + 25
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.grid(False)
    for i, (st, r) in enumerate(zip(steps, rasters)):
        t_off = i * (200 + gap)
        for key, _, color in order:
            m = np.isin(r[:, 0], list(rows[key]))
            ys = [rows[key][int(n)] for n in r[m, 0]]
            big = key in ("hit", "stand")
            ax.scatter(r[m, 1] + t_off, ys, s=30 if big else 2.5, marker="|" if big else ".",
                       color=color, linewidths=1.4 if big else 0, rasterized=True)
        s_, dealer, soft = st["obs"]
        action = "HIT" if st["action"] == 1 else "STAND"
        ax.text(t_off + 100, y0 + 5, f"sum {s_}{' (soft)' if soft else ''} vs dealer "
                f"{CARD.get(dealer, dealer)}\nHIT pool {st['n_hit']} spikes, STAND pool "
                f"{st['n_stand']}  →  {action}", ha="center", va="bottom", fontsize=8.5)
        ax.axvspan(t_off, t_off + 200, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([i * (200 + gap) + t for i in range(len(steps)) for t in (0, 100, 200)],
                  [str(t) for _ in steps for t in (0, 100, 200)])
    ax.set_xlabel("time within each 200 ms card-odor presentation (ms)")
    ax.set_ylim(-10, y0 + 60)
    ax.spines["left"].set_visible(False)
    outcome = {1.0: "fly wins", 0.0: "push", -1.0: "fly loses"}[meta["reward"]]
    ax.set_title(f"The spiking brain plays a hand: fly {_cards(meta['player'])} vs dealer "
                 f"{_cards(meta['dealer'])} ({outcome})", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "brain_plays_a_hand.png", dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    learning_curves()
    policy_heatmaps()
    kenyon_code()
    hand_raster()
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
