import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np


def parallelCoordinatesPlot(
    title, N, data, category, ynames, colors=None, category_names=None
):
    """
    A legend is added, if category_names is not None.

    :param title: The title of the plot.
    :param N: Number of data sets (i.e., lines).
    :param data: A list containing one array per parallel axis, each containing N data points.
    :param category: An array containing the category of each data set.
    :param category_names: Labels of the categories. Must have the same length as set(category).
    :param ynames: The labels of the parallel axes.
    :param colors: A colormap to use.
    :return:
    """

    fig, host = plt.subplots()

    # organize the data
    ys = np.dstack(data)[0]
    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins -= dys * 0.05  # add 5% padding below and above
    ymaxs += dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(ynames, fontsize=14)
    host.tick_params(axis="x", which="major", pad=7)
    host.spines["right"].set_visible(False)
    host.xaxis.tick_top()
    host.set_title(title, fontsize=18)

    if colors is None:
        colors = plt.cm.tab10.colors
    if category_names is not None:
        legend_handles = [None for _ in category_names]
    else:
        legend_handles = [None for _ in set(category)]
    for j in range(N):
        # to just draw straight lines between the axes:
        # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

        # create bezier curves
        # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
        #   at one third towards the next axis; the first and last axis have one less control vertex
        # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
        # y-coordinate: repeat every point three times, except the first and last only twice
        verts = list(
            zip(
                [
                    x
                    for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)
                ],
                np.repeat(zs[j, :], 3)[1:-1],
            )
        )
        # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path, facecolor="none", lw=1, edgecolor=colors[category[j] - 1]
        )
        legend_handles[category[j] - 1] = patch
        host.add_patch(patch)

        if category_names is not None:
            host.legend(
                legend_handles,
                category_names,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=len(category_names),
                fancybox=True,
                shadow=True,
            )

    plt.tight_layout()
    plt.show()


# Example usage
"""if __name__ == '__main__':

    ynames = ['P1', 'P2', 'P3', 'P4', 'P5']
    N1, N2, N3 = 10, 5, 8
    N = N1 + N2 + N3
    category = np.concatenate([np.full(N1, 1), np.full(N2, 2), np.full(N3, 3)])
    y1 = np.random.uniform(0, 10, N) + 7 * category
    y2 = np.sin(np.random.uniform(0, np.pi, N)) ** category
    y3 = np.random.binomial(300, 1 - category / 10, N)
    y4 = np.random.binomial(200, (category / 6) ** 1 / 3, N)
    y5 = np.random.uniform(0, 800, N)

    parallelCoordinatesPlot(ynames=ynames, data=[y1, y2, y3, y4, y5], category=category, N=N,
                            title='Parallel Coordinates Plot without Legend')

    parallelCoordinatesPlot(ynames=ynames, data=[y1, y2, y3, y4, y5], category=category, N=N,
                            title='Parallel Coordinates Plot with Legend', category_names=['Cat1', 'Cat2', 'Cat3'])"""

# Data
# ====
ynames = ["overall accuracy", "yes (f1, n=552)", "no (f1, n=338)", "maybe (f1, n=110)"]
N = 6
category = np.concatenate(
    [
        np.full(1, 1),
        np.full(1, 2),
        np.full(1, 3),
        np.full(1, 4),
        np.full(1, 5),
        np.full(1, 6),
    ]
)
overall_accuracy = [0.465, 0.487, 0.539, 0.274, 0.505, 0.56]
yes = [
    0.5635148042024832,
    0.6576168929110106,
    0.6755905511811023,
    0.41363636363636364,
    0.6347687400318979,
    0.6776729559748428,
]
maybe = [
    0.06363636363636363,
    0.07048458149779736,
    0.1407035175879397,
    0.17579250720461095,
    0.031007751937984496,
    0.017241379310344827,
]
no = [
    0.44718792866941015,
    0.19501133786848074,
    0.3622641509433962,
    0.14588235294117646,
    0.34146341463414637,
    0.41830065359477125,
]
data = [overall_accuracy, yes, no, maybe]
category_names = [
    "EIR",
    "Gemma-7b-it",
    "GPT-4-t",
    "GPT-3.5-t",
    "Llama3-8b",
    "Llama3-70b",
]

parallelCoordinatesPlot(
    title="LLM evaluations on Swe-PubMedQA-L (n=1000)",
    N=N,
    data=data,
    category=category,
    ynames=ynames,
    category_names=category_names,
)
