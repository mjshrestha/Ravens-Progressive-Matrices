import os, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops
from time import time

def solve2by2(problem):
    ACCURACY_THRES = 90
    start_time = time()
    problem_name = problem.name
    figures = problem.figures

    # for figureName in problem.figures:
    #     figure = problem.figures[figureName]
    #     img  = Image.open(figure.visualFilename)
    #     img.show()

    # if problem.name =='Basic Problem B-10':
    #     print()
    # else:
    #     return -1

    ques = ["A", "B", "C"]
    options = ["1", "2", "3", "4", "5", "6"]
    imgA = Image.open(problem.figures["A"].visualFilename)
    imgB = Image.open(problem.figures["B"].visualFilename)
    imgC = Image.open(problem.figures["C"].visualFilename)

    histA = imgA.histogram()
    histB = imgB.histogram()
    histC = imgC.histogram()

    min_hist = min(histA[0], histB[0], histC[0])
    max_hist = max(histA[0], histB[0], histC[0])

    # detect_edges(imgA)

    if (min_hist / max_hist) < 0.15:  # check histogram pct
        ACCURACY_THRES = 95

    # all the checks
    # checking similarity for all three images
    val1 = compare_similarity(imgA, imgB, ACCURACY_THRES)
    val2 = compare_similarity(imgA, imgC, ACCURACY_THRES)
    # first the simplest, check if it is the same image
    if val1[0] and val2[0]:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = compare_similarity(imgA, img, ACCURACY_THRES)
            if val[0]:
                return int(opts)

    # check if any two images are the same
    if val1[0]:  # imgA and imgB are same, so find same as imagC
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = compare_similarity(imgC, img, ACCURACY_THRES)
            if val[0]:
                return int(opts)

    if val2[0]:  # imgA and imgC are same, so find same as imagB
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = compare_similarity(imgB, img, ACCURACY_THRES)
            if val[0]:
                return int(opts)

    val3 = compare_similarity(imgB, imgC, ACCURACY_THRES)
    if val3[0]:  # imgB and imgC are same, so find same as imagA
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = compare_similarity(imgA, img, ACCURACY_THRES)
            if val[0]:
                return int(opts)

    #### check for a mirror
    val_mirror = check_mirror(imgA, imgB, ACCURACY_THRES)
    if val_mirror:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_mirror(imgC, img, ACCURACY_THRES)
            if val:
                return int(opts)
    val_mirror = check_mirror(imgA, imgC, ACCURACY_THRES)
    if val_mirror:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_mirror(imgB, img, ACCURACY_THRES)
            if val:
                return int(opts)
    val_mirror = check_mirror(imgB, imgC, ACCURACY_THRES)
    if val_mirror:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_mirror(imgA, img, ACCURACY_THRES)
            if val:
                return int(opts)

    ### check for flip
    val_flip = check_flip(imgA, imgB, ACCURACY_THRES)
    if val_flip:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_flip(imgC, img, ACCURACY_THRES)
            if val:
                return int(opts)

    val_flip = check_flip(imgA, imgC, ACCURACY_THRES)
    if val_flip:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_flip(imgB, img, ACCURACY_THRES)
            if val:
                return int(opts)

    val_flip = check_flip(imgB, imgC, ACCURACY_THRES)
    if val_flip:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_flip(imgA, img, ACCURACY_THRES)
            if val:
                return int(opts)

    # ***********************************
    # Not implemented
    # checking A-B&A-C vs B-C&C-D
    # val_3_check = check_three(imgA,imgB,imgC,problem,options,ACCURACY_THRES)
    # if(val_3_check[0]==0):
    #    return val_3_check[1]
    # ***********************************

    # checking if it is a rotated image
    val4 = checkTranspose(imgA, imgB, ACCURACY_THRES)
    if val4[0] == 0:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_same_transpose(imgC, img, val4[1], ACCURACY_THRES)
            if val == 0:
                return int(opts)

    [val5, ang5] = checkTranspose(imgA, imgC, ACCURACY_THRES)
    if val5 == 0:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_same_transpose(imgB, img, ang5, ACCURACY_THRES)
            if val == 0:
                return int(opts)

    [val6, ang6] = checkTranspose(imgB, imgC, ACCURACY_THRES)
    if val6 == 0:
        for opts in options:
            img = Image.open(problem.figures[opts].visualFilename)
            val = check_same_transpose(imgA, img, ang6, ACCURACY_THRES)
            if val == 0:
                return int(opts)

    ## check fill
    [val_edge, opt] = check_edges_fill_unfill(imgA, imgB, imgC, problem, options)
    if val_edge == 0:
        return opt

    # ###############################################################################
    ## check XOR
    [val_edge, opt] = check_xor_two(imgA, imgB, imgC, problem, options,
                                         ACCURACY_THRES)  # check XOR of A and B with XOR of C and option
    if val_edge == 0:
        return opt
    [val_edge, opt] = check_xor_two(imgA, imgC, imgB, problem, options,
                                         ACCURACY_THRES)  # check XOR of A and C with XOR of B and option
    if val_edge == 0:
        return opt
    [val_edge, opt] = check_xor_two(imgB, imgC, imgA, problem, options,
                                         ACCURACY_THRES)  # check XOR of C and B with XOR of A and option
    if val_edge == 0:
        return opt

    ###############################################################################
    print("--- %s seconds ---" % (time() - start_time))
    return -1


def compare_similarity(img1, img2, ACCURACY_THRES):
    diff = np.logical_xor(img1, img2)
    sum_diff = diff.sum()
    x, y = img1.size
    pct_sim = 100.0 - (sum_diff / (x * y) * 100.0)  # finding similarity in pct
    # pct_diff = abs(pct_diff)
    # print(pct_diff)
    similar = False
    if pct_sim >= ACCURACY_THRES:
        similar = True
    return similar, pct_sim
    # s = 0
    # for band_index, band in enumerate(img1.getbands()):
    #     m1 = numpy.array([p[band_index] for p in img1.getdata()]).reshape(*img1.size)
    #     m2 = numpy.array([p[band_index] for p in img2.getdata()]).reshape(*img2.size)
    #     s += numpy.sum(numpy.abs(m1 - m2))
    # return s


def check_mirror( img1, img2, ACCURACY_THRES):
    im = ImageOps.mirror(img1)
    res = compare_similarity(im, img2, ACCURACY_THRES)
    # print(res)
    if res[1] >= ACCURACY_THRES:
        return True


def check_flip( img1, img2, ACCURACY_THRES):
    im = ImageOps.flip(img1)
    res = compare_similarity(im, img2, ACCURACY_THRES)
    # print(res)
    if res[1] >= ACCURACY_THRES:
        return True

    # def check_three( img_A, img_B, img_C, problem, options, ACCURACY_THRES):
    #     A_B_sim = compare_similarity(img_A, img_B, ACCURACY_THRES)
    #     A_C_sim = compare_similarity(img_A, img_C, ACCURACY_THRES)
    #
    #     THRESHOLD = 5
    #     for opts in options:
    #         img = Image.open(problem.figures[opts].visualFilename)
    #         B_D_sim = compare_similarity(img_B, img, ACCURACY_THRES)
    #         C_D_sim = compare_similarity(img_C, img, ACCURACY_THRES)
    #         if abs(A_B_sim[1] - B_D_sim[1]) <= THRESHOLD and abs(A_C_sim[1] - C_D_sim[1]) <= THRESHOLD:
    #             return 0, int(opts)
    #     return -1, -1


def checkTranspose( img1, img2, ACCURACY_THRES):
    angles = [45, 90, 135, 180, 225, 270, 315]
    # angles = [90,180,270]
    # in the rotation, need to set background as white
    for angs in angles:
        # original image
        img = img1
        x, y = img1.size
        # converted to have an alpha layer
        im2 = img.convert('RGBA')
        # rotated image
        # rot = im2.rotate(angs, expand=1)
        rot = im2.rotate(angs)
        # a white image same size as rotated image

        fff = Image.new('RGBA', rot.size, (255,) * 4)
        # create a composite image using the alpha layer of rot as a mask
        out = Image.composite(rot, fff, rot)

        out = ImageOps.fit(out, (x, y), centering=(0, 0))

        # out.show()
        # save your work (converting back to mode='1' or whatever..)
        # out.convert(img.mode).save('test2.bmp')
        same = compare_similarity(out, img2, ACCURACY_THRES)
        if same[0]:
            return 0, angs
    return -1, -1


def check_same_transpose( img1, img2, angle, ACCURACY_THRES):
    # imTemp = img1.rotate(angle)
    # original image
    img = img1
    # converted to have an alpha layer
    im2 = img.convert('RGBA')
    # rotated image
    # rot = im2.rotate(angle, expand=1)
    rot = im2.rotate(angle)
    # a white image same size as rotated image
    fff = Image.new('RGBA', rot.size, (255,) * 4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(rot, fff, rot)
    # save your work (converting back to mode='1' or whatever..)
    # out.convert(img.mode).save('test2.bmp')
    same = compare_similarity(out, img2, ACCURACY_THRES)
    if same[0]:
        return 0


def check_edges( img1, img2):
    im1_edge = img1.filter(ImageFilter.FIND_EDGES)
    im2_edge = img2.filter(ImageFilter.FIND_EDGES)

    # im1_edge.show()
    # im2_edge.show()
    THRESHOLD_EDGES = 83
    diff_edges = compare_similarity(im1_edge, im2_edge, THRESHOLD_EDGES)
    # print(diff_edges)
    s2 = -1
    if diff_edges[0]:
        s2 = 0
    return s2


def check_edges_fill_unfill( imgA, imgB, imgC, problem, options):
    histA = imgA.histogram()
    histB = imgB.histogram()
    histC = imgC.histogram()

    min_hist = min(histA[0], histB[0], histC[0])
    max_hist = max(histA[0], histB[0], histC[0])

    hist_ratio_check = False
    HIST_THRESHOLD = 0.15
    if (min_hist / max_hist) < HIST_THRESHOLD:  # check histogram pct to see if filled an unfilled image is present
        hist_ratio_check = True
        val = check_edges(imgA, imgB)
        if val == 0:
            for opts in options:
                img = Image.open(problem.figures[opts].visualFilename)
                edge_check = check_edges(imgC, img)
                if check_fill_unfill(imgC, img) == 0:
                    if edge_check == 0:
                        return 0, int(opts)

        val = check_edges(imgA, imgC)
        if val == 0:
            for opts in options:
                img = Image.open(problem.figures[opts].visualFilename)
                edge_check = check_edges(imgB, img)
                if check_fill_unfill(imgB, img) == 0:
                    if edge_check == 0:
                        return 0, int(opts)

        val = check_edges(imgB, imgC)
        if val == 0:
            for opts in options:
                img = Image.open(problem.figures[opts].visualFilename)
                edge_check = check_edges(imgA, img)
                if check_fill_unfill(imgA, img) == 0:
                    if edge_check == 0:
                        return 0, int(opts)

    return -1, -1

    # first find image with same edge


def check_fill_unfill( img1, img2):
    hist1 = img1.histogram()
    hist2 = img2.histogram()
    maxH = max(hist1[0], hist2[0])
    minH = min(hist1[0], hist2[0])

    if maxH != 0 and (minH / maxH) < 0.15:
        return 0

    return -1


def check_xor_two( img1, img2, img3, problem, options, ACCURACY_THRES):  # check if two are xor
    # im = np.logical_xor(img1, img2)
    img1 = img1.convert("1")
    img2 = img2.convert("1")
    img3 = img3.convert("1")
    im = ImageChops.logical_xor(img1, img2)
    im = ImageChops.invert(im)
    #  print(im.size)
    #  print(img1.size)
    # res = compare_similarity(im, img2, ACCURACY_THRES)
    similarity = -1  # original best similarity value
    most_similar = -1  # original best similar

    for opts in options:
        img = Image.open(problem.figures[opts].visualFilename)
        img = img.convert("1")
        im2 = ImageChops.logical_xor(img, img3)
        im2 = ImageChops.invert(im2)
        # img = load_binary_image(problem.figures[opts].visualFilename)  # loading as binary image
        val = compare_similarity(im, im2, ACCURACY_THRES)
        if val[0]:
            if val[1] > similarity:
                similarity = val[1]
                most_similar = int(opts)
    return 0, most_similar
