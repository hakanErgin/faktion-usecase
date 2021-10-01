import cv2 as cv
import glob

# Courtesy https://leslietj.github.io/2020/06/28/How-to-Average-Images-Using-OpenCV/

for j in range(11):
    # import all image files with the .jpg extension
    images = glob.glob('assets/normal_dice/'+str(j)+'/*.jpg')

    image_data = []
    for img in images:
        this_image = cv.imread(img, 1)
        image_data.append(this_image)

    avg_image = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            avg_image = cv.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)

    cv.imwrite('assets/normal_dice/avg_'+str(j)+'.jpg', avg_image)
    # avg_image = cv.imread('assets/normal_dice/avg_'+str(j)+'.png')
    # plt.imshow(avg_image)
    # plt.show()

