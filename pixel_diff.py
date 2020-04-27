from diffimg import diff

def simple_pixel_diff(f1, f2, diff_img_file):
    return 1 - diff(f1, f2, diff_img_file=diff_img_file, ignore_alpha=False)
