import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def detect_features(img, show_image=True):
    """
    Detects features within an image.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, features = sift.detectAndCompute(gray_img, None)
    
    if show_image:
        sift_img = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("SIFT Keypoints", sift_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints, features


def crop(img):
    """
    Removes any black part of an image.
    """
    y_non, x_non, _ = np.nonzero(img)
    return img[min(y_non):max(y_non), min(x_non):max(x_non)]


def feature_match_and_image_alignment(img1, img2, keypoints1, features1, keypoints2, features2, show_img=True):
    """
    Establishes good matches between consecutive pairs of images.
    """
    bfmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    matches = bfmatcher.match(features1, features2)
    pts1 = [keypoints1[match.queryIdx].pt for match in matches]
    pts2 = [keypoints2[match.trainIdx].pt for match in matches]

    homography, mask = cv2.findHomography(np.array(pts2), np.array(pts1), cv2.RANSAC)
    warped_img1 = cv2.warpPerspective(img2, homography, (img1.shape[1]+img2.shape[1], img2.shape[0]))
    padded_img1 = cv2.copyMakeBorder(img1, 0, 0, 0, img2.shape[1], cv2.BORDER_CONSTANT)

    dist1 = get_distance_transform(padded_img1)
    dist2 = get_distance_transform(warped_img1)

    result = ((padded_img1 * dist1 + warped_img1 * dist2) / (np.maximum(dist1 + dist2, 1))).astype("uint8")

    return result


def get_distance_transform(img_rgb):
    """
    Get distance to the closest background pixel for an RGB image
    Input :
        img_rgb : np . array , HxWx3 RGB image
    Output :
        dist : np.array , HxWx1 distance image
        each pixel 's intensity is proportional to
        its distance to the closest background pixel
        scaled to [0..255] for plotting
    """
    thresh = cv2.threshold(img_rgb, 0, 255, cv2.THRESH_BINARY)[1]
    thresh = thresh.any(axis=2)
    thresh = np.pad(thresh, 1)

    dist = distance_transform_edt(thresh)[1: -1 , 1: -1]

    dist = dist [:, :, None]
    return dist / dist.max () * 255.0


def stich_n_images(images):
    """
    Stiches n images together
    """
    while len(images) > 1:
        for i in range(len(images) - 1):
            keypoints1, features1 = detect_features(images[i], show_image=False)
            keypoints2, features2 = detect_features(images[i+1], show_image=False)

            outimg = crop(feature_match_and_image_alignment(images[i], images[i+1], keypoints1, features1, keypoints2, features2))
            images[i] = outimg

        images.pop()

    return images[0]


def main():
    img_path1 = "images\yosemite1.jpg"
    img_path2 = "images\yosemite2.jpg"
    img_path3 = "images\yosemite3.jpg"
    img_path4 = "images\yosemite4.jpg"
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    img3 = cv2.imread(img_path3)
    img4 = cv2.imread(img_path4)

    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))
    img3 = cv2.resize(img3, (640, 480))
    img4 = cv2.resize(img4, (640, 480))

    images = [img1, img2, img3, img4]
    stiched_image = cv2.resize(stich_n_images(images), (1500, 480))
        
    cv2.imshow("output", stiched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()