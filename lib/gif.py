import contextlib
from pathlib import Path

from PIL import Image


def create_gif(src_dir, fp_out: str):

    if isinstance(fp_out, Path):
        # fp_out = fp_out.resolve()
        if not str(fp_out).endswith('.gif'):
            raise ValueError(f'Output file should end in ".gif" but got {fp_out}')
    elif isinstance(fp_out, str) and not fp_out.endswith('.gif'):
        raise ValueError(f'Output file should end in ".gif" but got {fp_out}')

    # use exit stack to automatically close opened images
    file_paths = sorted(list(Path(src_dir).glob('*.png')))
    if len(file_paths) == 0:
        raise FileNotFoundError('File path list is empty.')
    file_paths += [file_paths[-1] for _ in range(20)]
    with contextlib.ExitStack() as stack:
        imgs = (stack.enter_context(Image.open(f))
                for f in file_paths)
        img = next(imgs)
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=100, loop=0)
    print(f'Image saved at {fp_out}.')