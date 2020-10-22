import os, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops
from time import time

ques_img = []  # array to store images
# hist_img = []  # store image histrograms
ans_img = []  # array to store options for answers

best_ans = -1  # to save the best answer after checking every functions
best_diff = sys.maxsize  # to save the least difference value after every function


def solve3by3_till_C(problem):
    ACCURACY_THRES = 1.5
    global best_ans, best_diff
    global ques_img, hist_img, ans_img
    ques_img.clear()  # array to store images
    #    hist_img.clear()  # store image histrograms
    ans_img.clear()  # array to store options for answers
    best_ans = -1  # to save the best answer after checking every functions
    best_diff = sys.maxsize  # to save the least difference value after every function
    # start_time = time()
    # problem_name = problem.name
    # figures = problem.figures
    img_dim = Image.open(problem.figures["A"].visualFilename)  # reading the image to check the dimensions
    x, y = img_dim.size

    ques = ["A", "B", "C", "D", "E", "F", "G", "H"]
    options = ["1", "2", "3", "4", "5", "6", "7", "8"]
    min_hist = x * y
    max_hist = 0
    ######################################
    # if problem.name =='Basic Problem C-09' :#or problem.name =='Basic Problem C-05' :
    #     print()
    # else:
    #     return -1
    ######################################
    for nos in range(len(ques)):
        ques_img.append(Image.open(problem.figures[ques[nos]].visualFilename))
        ans_img.append(Image.open(problem.figures[options[nos]].visualFilename))

    ans = check_similars(ques_img, ans_img, ACCURACY_THRES) #check similars, mirrors or flips
    if ans[0]:
        return ans[1]
    check_increasing_pixelTypes(ACCURACY_THRES) #check types with increasing pixels
    if best_ans != -1:
        return best_ans
    res = check_half_mirrorflip(ACCURACY_THRES) #check types where image is split into half and their mirror or flip is present in other C or G
    if res[0]:
        return res[1]
    return best_ans


def check_similars(ques_img, ans_img, ACCURACY_THRES):
    global best_ans,best_diff
    ans = check_similarity(ques_img[0], ques_img[1], ques_img[2], ques_img[7],
                           ACCURACY_THRES)  # checking cases where horizontal images are same
    if ans[0]:  # if found similar, return answer
        if ans[2]<best_diff:
            best_ans = ans[1]
            best_diff = ans[2]
    ans = check_similarity(ques_img[0], ques_img[3], ques_img[6], ques_img[5],
                           ACCURACY_THRES)  # checking cases where vertical images are same
    if ans[0]:  # if found similar, return answer
        if ans[2] < best_diff:
            best_ans = ans[1]
            best_diff = ans[2]

    ans = check_mirror(ques_img[0], ques_img[2], ques_img[6],
                       ACCURACY_THRES)  # mirror of A is C then mirror of G is ans
    if ans[0]:  # if found similar, return answer
        if ans[2]<best_diff:
            best_ans = ans[1]
            best_diff = ans[2]

    ans = check_mirror(ques_img[0], ques_img[6], ques_img[2], ACCURACY_THRES)
    if ans[0]:  # if found similar, return answer
        if ans[2]<best_diff:
            best_ans = ans[1]
            best_diff = ans[2]

    ans = check_flip(ques_img[0], ques_img[2], ques_img[6], ACCURACY_THRES)  # flip of A is C then flip of G is ans
    if ans[0]:  # if found similar, return answer
        if ans[2] < best_diff:
            best_ans = ans[1]
            best_diff = ans[2]

    ans = check_flip(ques_img[0], ques_img[6], ques_img[2], ACCURACY_THRES)  # flip of A is G then flip of C is ans
    if ans[0]:  # if found similar, return answer
        if ans[2] < best_diff:
            best_ans = ans[1]
            best_diff = ans[2]

    if best_ans!=-1:
        return  True, best_ans
    else:
        return False, 0
    # check_mirror(ques_img[2], ques_img[6], ques_img[0], ACCURACY_THRES) #mirror of C is G then mirror of A is ans


def check_similarity(img1_q, img2_q, img3_q, test_img, ACCURACY_THRES):
    best_ans=-1
    best_diff=sys.maxsize
    global ques_img, hist_img, ans_img
    img1 = img1_q.convert('1')
    img2 = img2_q.convert('1')
    img3 = img3_q.convert('1')
    diff1 = np.logical_xor(img1, img2)
    m_diff1 = diff1.mean() * 100
    diff2 = np.logical_xor(img2, img3)
    m_diff2 = diff2.mean() * 100
    similar = False
    if m_diff1 < ACCURACY_THRES and m_diff2 < ACCURACY_THRES:
        diff_temp = find_similar(ans_img, test_img)
        if diff_temp[1] < best_diff:  # find the smallest difference from the options
            best_diff = diff_temp[1]
            best_ans = diff_temp[0]
            similar = True
    return similar, best_ans, best_diff
#

def find_similar(ans_arr, img):
    """
        Finds the similar answer from the option given
        :param ans_arr: The answer options lists.
        :param img: The choice for which similar is required
        :return: The best choice and the similarity image.
        """
    best_index = -1
    best_diff = sys.maxsize
    for nos in range(len(ans_arr)):
        diff = np.logical_xor(img, ans_arr[nos])
        diff_mean = diff.mean()
        if diff.mean() < best_diff:
            best_index = nos + 1  # need to add 1 as index starts from 0 but answers start from 1
            best_diff = diff_mean
    return best_index, best_diff


def check_mirror(img1, img2, test_img, ACCURACY_THRES):
    """
         Checks if the img1 is mirror of img2
        returns the best option of the test_img which is the mirror in the answers
         """

    global ques_img, hist_img, ans_img
    im = ImageOps.mirror(img1)
    dark_pixels_im = im.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_img2 = img2.histogram()[0]
    if dark_pixels_im == 0 or dark_pixels_img2 == 0:
        dark_pixels_im = 1
    if dark_pixels_img2 == 0:
        dark_pixels_img2 = 1
    if dark_pixels_img2 > dark_pixels_im:
        ratio = abs(dark_pixels_img2 / dark_pixels_im)
    else:
        ratio = abs(dark_pixels_im / dark_pixels_img2)
    maybesame = False
    if ratio > 0.9 and ratio < 1.1:
        maybesame = True
    diff = np.logical_xor(im, img2)
    mean_diff = diff.mean()
    pct_diff = mean_diff * 100  # get the difference in pct for convenience
    mirror = False
    best_ans=-1
    best_diff=sys.maxsize
    if maybesame and pct_diff <= ACCURACY_THRES:
        test_img_mirror = ImageOps.mirror(test_img)  # get the mirror of the testing image as well
        diff_temp = find_similar(ans_img, test_img_mirror)
        if diff_temp[1] < best_diff:  # compare with the global to find the best answer
            best_diff = pct_diff
            best_ans = diff_temp[0]
            mirror = True
    return mirror, best_ans, best_diff


def check_flip(img1, img2, test_img, ACCURACY_THRES):
    best_ans=-1
    best_diff=sys.maxsize
    global ques_img, hist_img, ans_img
    im = ImageOps.flip(img1)
    diff = np.logical_xor(im, img2)
    mean_diff = diff.mean()
    pct_diff = mean_diff * 100  # get the difference in pct for convenience
    mirror = False
    dark_pixels_im = im.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_img2 = img2.histogram()[0]
    if dark_pixels_im == 0 or dark_pixels_img2 == 0:
        dark_pixels_im = 1
    if dark_pixels_img2 == 0:
        dark_pixels_img2 = 1
    if dark_pixels_img2 > dark_pixels_im:
        ratio = abs(dark_pixels_img2 / dark_pixels_im)
    else:
        ratio = abs(dark_pixels_im / dark_pixels_img2)
    maybesame = False
    if ratio > 0.9 and ratio < 1.1:
        maybesame = True
    if maybesame and pct_diff <= ACCURACY_THRES:
        # if pct_diff <= ACCURACY_THRES:
        test_img_mirror = ImageOps.flip(test_img)  # get the mirror of the testing image as well
        diff_temp = find_similar(ans_img, test_img_mirror)
        if diff_temp[1] < best_diff:  # compare with the global to find the best answer
            best_diff = pct_diff
            best_ans = diff_temp[0]
            mirror = True
    return mirror, best_ans,best_diff


def check_increasing_pixelTypes(ACCURACY_THRES):
    selected_ans = -1
    global best_ans, best_diff
    global ques_img, hist_img, ans_img

    check_FH_same = check_increasing_size_FH_same(ACCURACY_THRES)
    if check_FH_same[1]:
        best_ans = check_FH_same[0]
    else:
        selected_ratio1 = check_increasing_rows()
        best_ans = selected_ratio1[0]


##########################################################
def find_substract(img1, img2):
    subs = ImageChops.subtract(img1, img2)
    subs = subs.convert('L')
    img3 = ImageOps.invert(subs)
    return img3


###########################################################

# check if the pixels is increasing from A to B to C, then find the same ratio as C/B with ans/H as the ratio is preserved
def check_increasing_rows():
    global best_ans, best_diff
    global ques_img, hist_img, ans_img
    dark_pA = ques_img[0].histogram()[0]  # A
    dark_pB = ques_img[1].histogram()[0]  # B
    dark_pC = ques_img[2].histogram()[0]  # C
    dark_pD = ques_img[3].histogram()[0]  # D
    dark_pE = ques_img[4].histogram()[0]  # E
    dark_pF = ques_img[5].histogram()[0]  # F
    dark_pG = ques_img[6].histogram()[0]  # G
    dark_pH = ques_img[7].histogram()[0]  # H

    if (dark_pA == 0):
        dark_pA = 1
    ratio_B_A = dark_pB / dark_pA
    ratio_C_B = dark_pC / dark_pB
    ratio_H_G = dark_pH / dark_pG
    ratio_F_E = dark_pF / dark_pE
    ratio_H_E = dark_pH / dark_pE
    ratio_G_D = dark_pG / dark_pD
    ratio_C_F = dark_pC / dark_pF  # for c-08
    ratio_G_H = dark_pG / dark_pH  # for c-08
    sel_ratio = -1
    diff = sys.maxsize
    sel_ans = -1
    avg_ansHF = sys.maxsize
    if (ratio_B_A > 1 and ratio_C_B > 1):  # pixels could be increasing or also decreasing
        for nos in range(len(ans_img)):  # check for solutions
            sel_dark_pix = ans_img[nos].histogram()[0]
            ratio_ans_H = sel_dark_pix / dark_pH
            ratio_ans_F = sel_dark_pix / dark_pF
            if ratio_ans_H > 1.05 and ratio_ans_F > 1.05:  # increasing pixels case
                diff_ans1 = abs(ratio_ans_H - ratio_C_B)
                diff_ans2 = abs(ratio_ans_F - ratio_H_E)
                avg_ansHF_new = (ratio_ans_H)
                if diff_ans1 < diff and avg_ansHF_new < avg_ansHF:  # the ratio of the answer to H is closest than others
                    diff = diff_ans1
                    sel_ans = nos + 1
                    sel_ratio = ratio_ans_H
                    avg_ansHF = avg_ansHF_new
    elif (ratio_B_A < 1 and ratio_C_B < 1):  # pixels could be increasing or also decreasing
        avg_CB_FE = (ratio_C_B + ratio_F_E) / 2  # to improve, we take average of the ratio of of CtoB and F to E
        for nos in range(len(ans_img)):
            sel_dark_pix = ans_img[nos].histogram()[0]
            ratio_ans_H = sel_dark_pix / dark_pH
            ratio_ans_F = sel_dark_pix / dark_pF
            if ratio_ans_H < 1.1 and ratio_ans_F < 1.1:  # decreasing pixels case
                diff_ans1 = abs(ratio_ans_H - ratio_C_B)
                avg_ansHF_new = (ratio_ans_H + ratio_ans_F) / 2
                if (diff_ans1 < diff and avg_ansHF_new < avg_ansHF):
                    diff = diff_ans1
                    sel_ans = nos + 1
                    sel_ratio = ratio_ans_H

    elif (ratio_G_H > 1 and ratio_C_F > 1):  # pixels could be increasing or also decreasing
        avg_CB_FE = (ratio_C_B + ratio_F_E) / 2  # to improve, we take average of the ratio of of CtoB and F to E
        for nos in range(len(ans_img)):
            sel_dark_pix = ans_img[nos].histogram()[0]
            ratio_ans_H = dark_pH / sel_dark_pix
            ratio_ans_F = dark_pF / sel_dark_pix

            if ratio_ans_H < 1 and ratio_ans_F < 1:  # decreasing pixels case
                diff_ans1 = abs(ratio_ans_H - ratio_ans_F)
                diff_ans2 = abs(ratio_ans_F - ratio_H_E)
                avg_ansHF_new = (ratio_ans_H - ratio_ans_F) / 2
                if (diff_ans1 < diff and diff_ans1 > 1.5):
                    diff = diff_ans1
                    sel_ans = nos + 1
                    sel_ratio = ratio_ans_H

    return sel_ans, sel_ratio, diff


# Here we check if increasing C and F and increasing pixels G and H then ans is also increasing
def check_increasing_size_FH_same(ACCURACY_THRES):
    global best_ans, best_diff
    global ques_img, hist_img, ans_img
    dark_pixels_A = ques_img[0].histogram()[0]
    dark_pixels_C = ques_img[2].histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_D = ques_img[3].histogram()[0]
    # dark_pixels_F = np.sum(img2)
    dark_pixels_F = ques_img[5].histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    # dark_pixels_G = np.sum(img3)
    dark_pixels_G = ques_img[6].histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_H = ques_img[7].histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    ratio = sys.maxsize
    sel_ans = -1
    diff = sys.maxsize
    thres = 0.1
    # check if ratio from C to F is similar to ratio from G to H
    ratio_F_C = dark_pixels_F / dark_pixels_C
    ratio_H_G = dark_pixels_H / dark_pixels_G
    diff_val = sys.maxsize
    sel_opt = -1
    select = False
    diff = np.logical_xor(ques_img[5], ques_img[7])  # check if F and H are same
    mean_diff = diff.mean()
    pct_diff = mean_diff * 100  # get the difference in pct for convenience
    if dark_pixels_H > 10000:  # if we have more dark pixels, then threhold value needs to be higher
        ACCURACY_THRES = 2.5
    if pct_diff < ACCURACY_THRES and dark_pixels_F > dark_pixels_C and dark_pixels_H > dark_pixels_G:
        if abs(ratio_F_C - ratio_H_G) <= thres:  # checking if they are the same within the threshold
            for nos in range(len(ans_img)):
                sel_dark_pix = ans_img[nos].histogram()[0]
                if sel_dark_pix > dark_pixels_F and sel_dark_pix > dark_pixels_H:
                    ratio_ans_F = sel_dark_pix / dark_pixels_F
                    ratio_ans_H = sel_dark_pix / dark_pixels_H
                    diff_ratio = abs(ratio_ans_F - ratio_ans_H)
                    if abs(ratio_ans_F - ratio_ans_H) < thres:  # if the ratio of option to F and H is the same then it could be the ans
                        if diff_ratio < diff_val:
                            diff_val = diff_ratio
                            sel_opt = nos + 1
                            select = True
    elif dark_pixels_F < dark_pixels_C and dark_pixels_H < dark_pixels_G:  # no of pixels could be decreasing
        if abs(ratio_F_C - ratio_H_G) <= thres:
            for nos in range(len(ans_img)):
                sel_dark_pix = ans_img[nos].histogram()[0]
                if sel_dark_pix < dark_pixels_F and sel_dark_pix < dark_pixels_H:
                    ratio_ans_F = sel_dark_pix / dark_pixels_F
                    ratio_ans_H = sel_dark_pix / dark_pixels_H
                    diff_ratio = abs(ratio_ans_F - ratio_ans_H)
                    if abs(
                            ratio_ans_F - ratio_ans_H) < thres:  # if the ratio of option to F and H is the same then it could be the ans
                        if diff_ratio < diff_val:
                            # if (diff_ratio!=0) & (diff_ratio<=thres):#and abs(diff_ratio-ratio_H_G) - (diff_ratio-ratio_F_C)<=thres:
                            diff_val = diff_ratio
                            sel_opt = nos + 1
                            select = True
    #########################################################################
    if dark_pixels_F < dark_pixels_C and dark_pixels_H < dark_pixels_G and dark_pixels_A < dark_pixels_D and dark_pixels_C > dark_pixels_F:  # no of pixels could be decreasing
        diff_val = sys.maxsize
        sel_opt = -1
        res = for_C08() # new code implemented
        if res[0]:
            select=True
            sel_opt=res[1]
    #########################################################################
    return sel_opt, select


def check_half_mirrorflip(ACCURACY_THRES):
    global best_ans, best_diff
    res = check_vert(ques_img[0], ques_img[2], ques_img[6], ACCURACY_THRES)
    if res[0]:
        if res[2] < best_diff:
            best_ans = res[1]
            best_diff = res[2]
    res = check_vert(ques_img[0], ques_img[6], ques_img[2], ACCURACY_THRES)
    if res[0]:
        if res[2] < best_diff:
            best_ans = res[1]
            best_diff = res[2]
    res = check_horiz(ques_img[0], ques_img[2], ques_img[6], ACCURACY_THRES)
    if res[0]:
        if res[2] < best_diff:
            best_ans = res[1]
            best_diff = res[2]

    res = check_horiz(ques_img[0], ques_img[6], ques_img[2], ACCURACY_THRES)
    if res[0]:
        if res[2] < best_diff:
            best_ans = res[1]
            best_diff = res[2]

    if best_ans!=-1:
        return True, best_ans, best_diff
    else:
        return False, -1,-1

def check_vert(img1, img2, img3, ACCURACY_THRES):
    #global best_ans, best_diff
    global ques_img, hist_img, ans_img
    im_left_A, im_left_C, im_right_A, im_right_C = crop_image(img1, img2, how="vertical")

    im_left_A_mirror = ImageOps.mirror(im_left_A)
    im_right_A_mirror = ImageOps.mirror(im_right_A)

    im_left_A_flip = ImageOps.flip(im_left_A)
    im_right_A_flip = ImageOps.flip(im_right_A)
    # testing for mirror
    res_left_mirror = check_twoimages_same(im_left_A_mirror, im_left_C, ACCURACY_THRES)
    res_right_mirror = check_twoimages_same(im_right_A_mirror, im_right_C, ACCURACY_THRES)
    # testing for flips
    res_left_flip = check_twoimages_same(im_left_A_flip, im_left_C, ACCURACY_THRES)
    res_right_flip = check_twoimages_same(im_right_A_flip, im_right_C, ACCURACY_THRES)
    found_ans = False
    best_ans_here = -1
    diff_min = sys.maxsize
    if res_left_mirror[0] and res_right_mirror[0]:
        # let us find the half of G as it will help us find the ans
        img_G = img3
        [row_G, col_G] = img_G.size
        im_left_G = img_G.crop((0, 0, col_G / 2, row_G))  # left image from the middle
        im_right_G = img_G.crop((col_G / 2 + 1, 0, col_G, row_G))  # right image from the middle
        im_left_G_mirror = ImageOps.mirror(im_left_G)
        im_right_G_mirror = ImageOps.mirror(im_right_G)
        # im_right_G_mirror.show()
        for nos in range(len(ans_img)):
            opt_img = ans_img[nos]
            [row_opt, col_opt] = opt_img.size
            im_left_opt = opt_img.crop((0, 0, col_opt / 2, row_opt))  # left image from the middle
            # im_left_opt.show()
            im_right_opt = opt_img.crop((col_opt / 2 + 1, 0, col_opt, row_opt))  # right image from the middle
            res_opt_left = check_twoimages_same(im_left_G_mirror, im_left_opt, ACCURACY_THRES)
            res_opt_right = check_twoimages_same(im_right_G_mirror, im_right_opt, ACCURACY_THRES)
            if res_opt_left[0] and res_opt_right[0]:
                if res_opt_left[1] < diff_min:
                    diff_min = res_opt_left[1]
                    best_ans_here = nos + 1
                    found_ans = True
    elif res_left_flip[0] and res_right_flip[0]:  # here the image is flipped half or A in C
        # let us find the half of G as it will help us find the ans
        img_G = img3
        [row_G, col_G] = img_G.size
        im_left_G = img_G.crop((0, 0, col_G / 2, row_G))  # left image from the middle
        im_right_G = img_G.crop((col_G / 2 + 1, 0, col_G, row_G))  # right image from the middle
        im_left_G_flip = ImageOps.flip(im_left_G)
        im_right_G_flip = ImageOps.flip(im_right_G)
        # im_right_G_mirror.show()
        for nos in range(len(ans_img)):
            opt_img = ans_img[nos]
            [row_opt, col_opt] = opt_img.size
            im_left_opt = opt_img.crop((0, 0, col_opt / 2, row_opt))  # left image from the middle
            # im_left_opt.show()
            im_right_opt = opt_img.crop((col_opt / 2 + 1, 0, col_opt, row_opt))  # right image from the middle
            res_opt_left = check_twoimages_same(im_left_G_flip, im_left_opt, ACCURACY_THRES)
            res_opt_right = check_twoimages_same(im_right_G_flip, im_right_opt, ACCURACY_THRES)
            if res_opt_left[0] and res_opt_right[0]:
                if res_opt_left[1] < diff_min:
                    diff_min = res_opt_left[1]
                    best_ans_here = nos + 1
                    found_ans = True

    return found_ans, best_ans_here, diff_min
#########################################################################
def for_C08():
    global ques_img, ans_img
    imgA = ques_img[0].convert("1")
    imgB = ques_img[1].convert("1")
    imgC = ques_img[2].convert("1")
    imgD = ques_img[3].convert("1")
    imgE = ques_img[4].convert("1")
    imgF = ques_img[5].convert("1")
    imgG = ques_img[6].convert("1")
    imgH = ques_img[7].convert("1")

    dark_pixels_A = imgA.histogram()[0]
    dark_pixels_B = imgB.histogram()[0]
    dark_pixels_C = imgC.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_D = imgD.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_E = imgE.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_F = imgF.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_G = imgG.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_H = imgH.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    found =False
    if dark_pixels_B>dark_pixels_A and dark_pixels_C>dark_pixels_B and dark_pixels_D>dark_pixels_A and dark_pixels_G>dark_pixels_D:
        imgE_split = crop_image_single(imgE)# get the pixel values in each split squares
        dark_E = get_darkPixels(imgE_split)
        imgD_split = crop_image_single(imgD)
        dark_D = get_darkPixels(imgD_split)
        imgB_split = crop_image_single(imgB)
        dark_B = get_darkPixels(imgB_split)
        #for i in range(len(dark_E)):
        check_E = [pdcts(dark_B[0],dark_D[0]),pdcts(dark_B[1],dark_D[1]),pdcts(dark_B[2],dark_D[2]),pdcts(dark_B[3],dark_D[3])]
        if check_E == dark_E: # the condition is true, now we need to find blocks for C and G
            imgC_split = crop_image_single(imgC)
            dark_C = get_darkPixels(imgC_split)
            imgG_split = crop_image_single(imgG)
            dark_G = get_darkPixels(imgG_split)
            check_find = [pdcts(dark_C[0],dark_G[0]),pdcts(dark_C[1],dark_G[1]),pdcts(dark_C[2],dark_G[2]),pdcts(dark_C[3],dark_G[3])]
            for nos in range(len(ans_img)):
                opt = ans_img[nos]
                opt_split =  crop_image_single(opt)
                dark_opt = get_darkPixels(opt_split)
                if check_find==dark_opt:
                    found =True
                    return found, nos+1
    return found, -1


#########################################################################
def pdcts(a,b):
    ans=-2
    if (a==-1 and b==-1) or (b==-1 and a ==-1):
        ans=-1
    elif (a==1 and b ==1) or(b == 1 and a == 1):
        ans = 0
    elif (a ==1 and b ==-1) or(b ==1 and a ==-1):
        ans=1
    elif (a==0 and b ==0) or (b==0 and a==0):
        ans = 0
    return ans
#########################################################################
def get_darkPixels(img_list):
    dark_pix = [0,0,0,0]
    dark_pix[0] = img_list[0].histogram()[0]
    dark_pix[1] = img_list[1].histogram()[0]
    dark_pix[2] = img_list[2].histogram()[0]
    dark_pix[3] = img_list[3].histogram()[0]
    val=[-2,-2,-2,-2]
    for i in range(len(dark_pix)):
        pix = dark_pix[i]
        if pix > 1500 and pix < 3000:
            val[i] = 1
        elif pix>200 and pix < 1000:
            val[i] = 0
        elif pix<100:
            val[i]=-1

    return val    #im_left_t, im_left_b, im_right_t, im_right_b

#########################################################################
def crop_image_single(img):
    [row_A, col_A] = img.size
    im_left_t = img.crop((0, 0, col_A / 2, row_A / 2))  # left image from the middle
    im_left_b = img.crop((0, row_A / 2 + 1, col_A / 2, row_A))
    im_right_t = img.crop((col_A / 2 + 1, 0, col_A, row_A / 2))  # left image from the middle
    im_right_b = img.crop((col_A / 2 + 1, row_A / 2 + 1, col_A, row_A))

    return im_left_t, im_left_b, im_right_t, im_right_b
#########################################################################

def crop_image(img1, img2, how="vertical"):
    img_A = img1
    img_C = img2
    # Setting the points for cropped image
    [row_A, col_A] = img_A.size
    [row_C, col_C] = img_C.size
    if how == "vertical":
        im_left_A = img_A.crop((0, 0, col_A / 2, row_A))  # left image from the middle
        im_right_A = img_A.crop((col_A / 2 + 1, 0, col_A, row_A))  # right image from the middle
        im_left_C = img_C.crop((0, 0, col_C / 2, row_C))  # left image from the middle
        im_right_C = img_C.crop((col_C / 2 + 1, 0, col_C, row_C))  # right image from the middle
#########################################################################
    else:
        im_left_A = img_A.crop((0, 0, col_A, row_A / 2))  # left image from the middle
        im_right_A = img_A.crop((0, row_A / 2 + 1, col_A, row_A))  # right image from the middle
        im_left_C = img_C.crop((0, 0, col_C, row_C / 2))  # left image from the middle
        im_right_C = img_C.crop((0, row_C / 2 + 1, col_C, row_C))  # right image from the middle
#########################################################################
    return im_left_A, im_left_C, im_right_A, im_right_C


def check_horiz(img1, img2, img3, ACCURACY_THRES):
    global best_ans, best_diff
    global ques_img, hist_img, ans_img
    img_A = img1
    img_C = img2
    # Setting the points for cropped image
    [len_A, height_A] = img_A.size
    [len_C, height_C] = img_C.size

    im_top_A = img_A.crop((0, 0, len_A, height_A / 2))  # left image from the middle
    im_bottom_A = img_A.crop((0, height_A / 2, len_A, height_A))  # right image from the middle
    im_top_C = img_C.crop((0, 0, len_A, height_A / 2))  # left image from the middle
    im_bottom_C = img_C.crop((0, height_A / 2, len_A, height_A))  # right image from the middle

    im_top_A_mirror = ImageOps.mirror(im_top_A)
    im_bottom_A_mirror = ImageOps.mirror(im_bottom_A)

    im_top_A_flip = ImageOps.flip(im_top_A)
    im_bottom_A_flip = ImageOps.flip(im_bottom_A)
    # testing for mirror
    res_top_mirror = check_twoimages_same(im_top_A_mirror, im_top_C, ACCURACY_THRES)
    res_bottom_mirror = check_twoimages_same(im_bottom_A_mirror, im_bottom_C, ACCURACY_THRES)
    # testing for flips
    res_left_flip = check_twoimages_same(im_top_A_flip, im_top_C, ACCURACY_THRES)
    res_right_flip = check_twoimages_same(im_bottom_A_flip, im_bottom_C, ACCURACY_THRES)
    found_ans = False
    best_ans_here = -1
    diff_min = sys.maxsize
    if res_top_mirror[0] and res_bottom_mirror[0]:
        # let us find the half of G as it will help us find the ans
        img_C = img3
        [len_C, height_C] = img_C.size
        im_top_C = img_C.crop((0, 0, len_A, height_A / 2))  # top image from the middle
        im_bottom_C = img_C.crop((0, height_A / 2, len_A, height_A))  # right image from the middle
        im_top_G_mirror = ImageOps.mirror(im_top_C)
        im_bottom_G_mirror = ImageOps.mirror(im_bottom_C)
        # im_bottom_G_mirror.show()
        for nos in range(len(ans_img)):
            opt_img = ans_img[nos]
            [row_opt, col_opt] = opt_img.size
            im_top_opt = opt_img.crop((0, height_A / 2, len_A, height_A))  # left image from the middle
            # im_top_opt.show()
            im_bottom_opt = opt_img.crop((0, height_A / 2, len_A, height_A))  # right image from the middle
            res_opt_top = check_twoimages_same(im_top_G_mirror, im_top_opt, ACCURACY_THRES)
            res_opt_bottom = check_twoimages_same(im_bottom_G_mirror, im_bottom_opt, ACCURACY_THRES)
            if res_opt_top[0] and res_opt_bottom[0]:
                if res_opt_top[1] < diff_min:
                    diff_min = res_opt_top[1]
                    best_ans_here = nos + 1
                    found_ans = True
    elif res_left_flip[0] and res_right_flip[0]:  # here the image is flipped half or A in C
        # let us find the half of G as it will help us find the ans
        img_C = img3
        [len_C, height_C] = img_C.size
        im_top_C = img_C.crop((0, height_A / 2, len_A, height_A))  # left image from the middle
        im_bottom_C = img_C.crop((0, height_A / 2, len_A, height_A))  # right image from the middle
        im_top_G_flip = ImageOps.flip(im_top_C)
        im_bottom_G_flip = ImageOps.flip(im_bottom_C)
        # im_bottom_G_mirror.show()
        for nos in range(len(ans_img)):
            opt_img = ans_img[nos]
            [row_opt, col_opt] = opt_img.size
            im_top_opt = opt_img.crop((0, 0, col_opt / 2, row_opt))  # left image from the middle
            # im_top_opt.show()
            im_bottom_opt = opt_img.crop((col_opt / 2 + 1, 0, col_opt, row_opt))  # right image from the middle
            res_opt_top = check_twoimages_same(im_top_G_flip, im_top_opt, ACCURACY_THRES)
            res_opt_bottom = check_twoimages_same(im_bottom_G_flip, im_bottom_opt, ACCURACY_THRES)
            if res_opt_top[0] and res_opt_bottom[0]:
                if res_opt_top[1] < diff_min:
                    diff_min = res_opt_top[1]
                    best_ans_here = nos + 1
                    found_ans = True

    return found_ans, diff_min,best_ans_here


def check_twoimages_same(img1, img2, ACCURACY_THRES):  # when checking if two images are same, we convert it to binary
    img1_bin = img1.convert('1')
    img2_bin = img2.convert('1')
    diff = np.bitwise_xor(img1_bin, img2_bin)
    # img1.show()
    # img2.show()
    diff_mean = diff.mean()
    if diff_mean < ACCURACY_THRES:
        return True, diff_mean
    return False, sys.maxsize  # else return false and high difference


def check_C08types(ACCURACY_THRES):
    global ques_img, ans_img
    imgA = ques_img[0].convert("1")
    imgB = ques_img[1].convert("1")
    imgC = ques_img[2].convert("1")
    imgD = ques_img[3].convert("1")
    imgE = ques_img[4].convert("1")
    imgF = ques_img[5].convert("1")
    imgG = ques_img[6].convert("1")
    imgH = ques_img[7].convert("1")

    dark_pixels_A = imgA.histogram()[0]
    dark_pixels_B = imgB.histogram()[0]
    dark_pixels_C = imgC.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_D = imgD.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_E = imgE.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_F = imgF.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_G = imgG.histogram()[0]  # get the number of dark pixels by getting the 0 intensity
    dark_pixels_H = imgH.histogram()[0]  # get the number of dark pixels by getting the 0 intensity

    xor_B_D = ImageChops.logical_xor(imgC, imgG)
    xor_B_D = xor_B_D.convert('L')
    xor_B_D = ImageOps.invert(xor_B_D)
    xor_B_D = xor_B_D.convert('1')
    # xor_B_D.show()
    ratio = -1
    best = -1
    found = False
    ratio_G_H = dark_pixels_G / dark_pixels_H
    if dark_pixels_B > dark_pixels_A and dark_pixels_C > dark_pixels_B and dark_pixels_G > dark_pixels_D and dark_pixels_D > dark_pixels_A:
        if dark_pixels_G > 1000:  # if there are more pixels, then we need higher threshold
            ACCURACY_THRES = 2.5
        res = check_twoimages_same(imgC, imgG, ACCURACY_THRES)
        if res[0]:
            for nos in range(len(ans_img)):
                opt_img = ans_img[nos].convert("1")
                opt_img_darkpixels = opt_img.histogram()[0]
                # ratio_opt_H = dark_pixels_H/opt_img_darkpixels
                if opt_img_darkpixels < dark_pixels_H and opt_img_darkpixels < dark_pixels_F:
                    if opt_img_darkpixels == 0:
                        opt_img_darkpixels = 1
                    ratio_H_ans = dark_pixels_H / opt_img_darkpixels
                    ratio_H_ans = dark_pixels_H / opt_img_darkpixels
                    print(ratio_H_ans)
                    if opt_img_darkpixels > ratio:
                        ratio = opt_img_darkpixels
                        best = nos + 1
                        Found = True
    return Found, best


def check_twoimages_same(img1, img2, ACCURACY_THRES):  # when checking if two images are same, we convert it to binary
    img1_bin = img1.convert('1')
    img2_bin = img2.convert('1')
    diff = np.bitwise_xor(img1_bin, img2_bin)
    # img1.show()
    # img2.show()
    diff_mean = diff.mean()
    if diff_mean < ACCURACY_THRES:
        return True, diff_mean
    return False, sys.maxsize  # else return false and high difference

