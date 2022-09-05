import matplotlib.pyplot as plt

COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(pil_img, model, prob, boxes):
    """Plot  the results from object detection inference."""
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                color=c,
                linewidth=3,
            )
        )
        cl = p.argmax()
        text = f"{model.config.id2label[cl.item()]}: {p[cl]:0.2f}"
        ax.text(
            xmin,
            ymin,
            text,
            fontsize=15,
            bbox=dict(facecolor="yellow", alpha=0.5),
        )
    plt.axis("off")
    plt.show()


def visualize_attention(
    img, model, probas, keep, conv_features, bboxes_scaled, dec_attn_weights
):
    # get the feature map shape
    h, w = conv_features[-1][0].shape[-2:]
    fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
    # colors = COLORS * 100
    for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(
        keep.nonzero(), axs.T, bboxes_scaled
    ):
        ax = ax_i[0]
        ax.imshow(dec_attn_weights[0, idx].view(h, w))
        ax.axis("off")
        ax.set_title(f"query id: {idx.item()}")
        ax = ax_i[1]
        ax.imshow(img)
        xmin, ymin, xmax, ymax = map(
            lambda x: x.detach(), (xmin, ymin, xmax, ymax)
        )
        ax.add_patch(
            plt.Rectangle(
                xy=(xmin, ymin),
                width=xmax - xmin,
                height=ymax - ymin,
                fill=False,
                color="blue",
                linewidth=3,
            )
        )
        ax.axis("off")
        ax.set_title(model.config.id2label[probas[idx].argmax().item()])
    fig.tight_layout()
