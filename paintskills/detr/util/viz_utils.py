import matplotlib.pyplot as plt
from PIL import Image
import io

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]    


# edited from https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=bawoKsBC7oBh
def plot_results(pil_img, boxes, captions, fontsize=25, figsize=(10,10), colors=None, linewidth=5):
    """
    pil_img: PIL Image class
    boxes: xyxy boxes (normalized)
    captions: box captions
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(pil_img)
    ax = plt.gca()

    img_w = pil_img.width
    img_h = pil_img.height
            
    for i in range(len(captions)):
        
        # normalized
        xmin, ymin, xmax, ymax = boxes[i]
        
        xmin *= img_w
        ymin *= img_h
        xmax *= img_w
        ymax *= img_h
        
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, img_w)
        ymax = min(ymax, img_h)
        
        caption = captions[i]

        if colors is not None:
            c = colors[i]
            # Case-insensitive X11/CSS4 color name with no spaces.
            # https://en.wikipedia.org/wiki/X11_color_names
            # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        else:
            c = COLORS[i % len(COLORS)]
        
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=linewidth))
        text = caption
        ax.text(xmin, ymin, text, fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.close()
    return fig

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img

def show_images(imgs, gray=False, w=2, h=2, title=None):
    """Horizontally plot pillow images
    w / h: single img size to show -> Final W/H: (w * len(imgs), h)
    gray: whether to use colormap gray
    """
    
    fig = plt.figure(figsize=(w * len(imgs), h))
    for i, img in enumerate(imgs):
        rows = 1
        columns = len(imgs)
        fig.add_subplot(rows, columns, i+1)

        if gray:
            plt.imshow(img, cmap=plt.cm.gray)
        else:
            plt.imshow(img)
        
        plt.axis('off')
        if title is not None:
            plt.title(title[i])
        else:
            plt.title(f"Iter: {i+1}")

    return fig