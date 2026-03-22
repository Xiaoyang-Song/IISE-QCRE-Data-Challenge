import os 
import cv2
import numpy as np

def fire_the_hole(img, blur_kernel_size=11, adaptive_thres_blocksize=11, adaptive_thres_C=-2,
                  morph_open_kernel_size=3, morph_close_kernel_size=23, erode_kernel_size=23):
    
    # Blur the image to mitigate the checkerboard
    img_blur = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    
    # Edge detection
    thresh = cv2.adaptiveThreshold(img_blur.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, adaptive_thres_blocksize, adaptive_thres_C)

    # Reduce the disconncted edges
    open_kernel = np.ones((morph_open_kernel_size, morph_open_kernel_size), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # filter out some lines
    # TODO: neccessary? 
    thresh = cv2.distanceTransform(thresh, cv2.DIST_L1, 0)
    ret, thresh = cv2.threshold(thresh.astype(np.uint8), 2, 255, cv2.THRESH_BINARY)

    # Bridge the gaps between holes
    close_kernel = np.ones((morph_close_kernel_size, morph_close_kernel_size), np.uint8)
    fused = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

    # Erode the connected lines
    erode_kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    fused = cv2.erode(fused, erode_kernel, iterations=1)
    
    # # For debug purpose
    # plt.imshow(fused, cmap='gray')
    # plt.show()

    # Build blocks
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fused, connectivity=8, ltype=cv2.CV_32S)
    found_anchors = []
    for i in range(1, num_labels):       
        # Extract the stats for this specific blob
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        pixel_area = stats[i, cv2.CC_STAT_AREA]
        # print(pixel_area)

        # Filter criterion
        if w < 230 or w > 420: continue
        if h < 90 or h > 270: continue
        if w / h > 3: continue
        if pixel_area/(w*h) < 0.5: continue

        found_anchors.append((x, y, w, h))
    
    # print(len(found_anchors))
    return found_anchors

def crop_image(img, x, y, w, h, width=384, height=256):
    center_x = int(x + w // 2)
    center_y = int(y + h // 2)
    # crop image
    left_x, right_x = center_x - 192, center_x + 192
    top_y, bot_y = center_y - 128, center_y + 128
    img_cropped = img[top_y:bot_y, left_x:right_x]
    return img_cropped

def image2crops(img, **params):
    """
    Crop the input image based on perfortions, output the cropped images.
    
    Args:
        img (np.ndarray) : Input image     
        params : Parameters for algorithm
    
    Returns:
        outputs (List(np.ndarray)) : List of the cropped images
    """
    anchors = fire_the_hole(img, **params)
    outputs = []
    for anchor in anchors:
        crop = crop_image(img, *anchor)
        outputs.append(crop)
    
    return outputs


if __name__ == '__main__':
    
    file_dir = 'Data/train_labeled/Labeled Images/20260215_000002_135937_combined.jpg'
    img = cv2.imread(file_dir, 0)
    res = image2crops(img)
    for i, crop in enumerate(res):
        cv2.imshow(f'crop {i+1}', crop)
        cv2.waitKey(0)
    cv2.destroyAllWindows()