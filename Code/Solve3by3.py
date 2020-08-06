import os, sys
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops
from time import time

ques_img = []  # array to store images
# hist_img = []  # store image histrograms
ans_img = []  # array to store options for answers

best_ans = -1  # to save the best answer after checking every functions
best_diff = sys.maxsize  # to save the least difference value after every function
imgA=[]
imgB=[]
imgC=[]
imgD=[]
imgE=[]
imgF=[]
imgG=[]
imgH=[]


def solve3by3(problem):
    ACCURACY_THRES = 4
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
    # #####################################
    # if problem.name =='Basic Problem E-12' :#or problem.name =='Basic Problem C-05' :
    #     print()
    # else:
    #     return -1
    ####################################
    for nos in range(len(ques)):
        ques_img.append(Image.open(problem.figures[ques[nos]].visualFilename))
        ans_img.append(Image.open(problem.figures[options[nos]].visualFilename))

    #convert all question and answer images to binary
    preprocessing()
    #DPR_IPR()
    # ans = remove_boundary(ACCURACY_THRES)
    # if ans[0]:
    #     return ans[1]
    bestdiff = sys.maxsize
    bestans = -1
    found = False
    ans = check_similars(ques_img, ans_img, ACCURACY_THRES) #check similars, mirrors or flips
    if ans[0]:
        bestdiff = ans[2]
        bestans =ans[1]
        #return ans[1]

    # ans = check_center_same(ACCURACY_THRES)
    # if ans[0]:
    #     return ans[1]

    # ans = third_is_result(ACCURACY_THRES)
    # if ans[0]:
    #     return ans[1]
    ans = when_addedrows_same(ACCURACY_THRES)
    if ans[0] and ans[1]<bestdiff:
        bestdiff = ans[2]
        bestans = ans[1]
       # return ans[1]
    # ans = check_halfcombined(ACCURACY_THRES)
    # if ans[0]:
    #     return ans[1]

    ans = using_regiongrowing(ACCURACY_THRES)
    if ans[0] and ans[1]<bestdiff:
        bestdiff = ans[2]
        bestans = ans[1]

    ans = e12Kinds()
    if ans[0] and ans[1]<bestdiff:
        bestdiff = ans[2]
        bestans = ans[1]
    if bestans != -1:
        return bestans

    check_increasing_pixelTypes(ACCURACY_THRES) #check types with increasing pixels

    if best_ans != -1 and best_diff<bestdiff:
        return best_ans
    res = check_half_mirrorflip(ACCURACY_THRES) #check types where image is split into half and their mirror or flip is present in other C or G
    if res[0]:
        return res[1]

    return best_ans

def preprocessing():
    #convert the images to binary for consitency
    global ques_img,ans_img
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH

    for i in range(len(ques_img)):
        ques_img[i]= ques_img[i].convert('1')
    for i in range(len(ans_img)):
        ans_img[i]= ans_img[i].convert('1')
    imgA = ques_img[0]
    imgB = ques_img[1]
    imgC = ques_img[2]
    imgD = ques_img[3]
    imgE = ques_img[4]
    imgF = ques_img[5]
    imgG = ques_img[6]
    imgH = ques_img[7]


def DPR_IPR():

    global ques_img, ans_img
    check_DPR_IPR(ques_img[0],ques_img[1],ques_img[2],ques_img[7])

def check_DPR_IPR(img1,img2,img3,img_check):
    [row, col] = img1.size
    img1_hist = img1.histogram()[0]
    img2_hist = img2.histogram()[0]
    img3_hist = img3.histogram()[0]

    img1_pct = img1_hist/(row*col) # #percentage = (binarized image dark pixel count) / (image total pixel count)
    img2_pct = img2_hist / (row * col)  # #percentage = (binarized image dark pixel count) / (image total pixel count)
    img3_pct = img3_hist / (row * col ) # #percentage = (binarized image dark pixel count) / (image total pixel count)

    dpr1_2 = abs(img1_pct-img2_pct)
    dpr2_3 = abs(img2_pct-img3_pct)

    and_1_2 = ImageChops.logical_and(img1,img2)
    or_1_2 = ImageChops.logical_or(img1,img2)
    ipr_1_2 = and_1_2/or_1_2
    print()

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
    ans = diagonalSame(ACCURACY_THRES)  # checking cases where diagonal images are same
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
        return  True, best_ans, best_diff
    else:
        return False, 0
    # check_mirror(ques_img[2], ques_img[6], ques_img[0], ACCURACY_THRES) #mirror of C is G then mirror of A is ans
# def xoring(THRESHOLD):
#     global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
#     xor_AB = ImageChops.logical_xor(imgA, imgB)
#     xor_AB = ImageChops.invert(xor_AB).convert("1")
#     same  = similaritycheck_newThreshold(xor_AB,imgC, THRESHOLD)
#     bestdiff = sys.maxsize
#     bestans = -1
#     found = False
#     if same[0]:
#         xor_GH = ImageChops.logical_xor(imgG, imgH)
#         xor_GH = ImageChops.invert(xor_GH).convert("1")
#         for no in range(len(ans_img)):
#             imgOpt = ans_img[no]
#             imgOpt = imgOpt.convert("1")
#             #xor_GH.show()
#             same_Opt = similaritycheck_newThreshold(imgOpt, xor_GH, THRESHOLD)
#             same_Opt_mean = same_Opt[1]
#             if same_Opt[0] and same_Opt_mean < bestdiff:
#                 #imgOpt.show()
#                 bestdiff = same_Opt_mean
#                 bestans = no + 1
#                 found = True
#         return found, bestans, bestdiff
#     xor_AD = ImageChops.logical_xor(imgA, imgD)
#     xor_AD = ImageChops.invert(xor_AD).convert("1")
#     same = similaritycheck_newThreshold(xor_AD, imgG, THRESHOLD)
#     bestdiff = sys.maxsize
#     bestans = -1
#     found = False
#     if same[0]:
#         xor_CF = ImageChops.logical_xor(imgC, imgF)
#         xor_CF = ImageChops.invert(xor_CF).convert("1")
#         for no in range(len(ans_img)):
#             imgOpt = ans_img[no]
#             imgOpt = imgOpt.convert("1")
#             same_Opt = similaritycheck_newThreshold(imgOpt, xor_CF, THRESHOLD)
#             same_Opt_mean = same_Opt[1]
#             if same_Opt[0] and same_Opt_mean < bestdiff:
#                 bestdiff = same_Opt_mean
#                 bestans = no + 1
#                 found = True
#         return found, bestans, bestdiff
#     return found, bestans, bestdiff
def region_growing(img):
    im = img.convert('1')
    [width, height] = im.size
    minX=width
    minY=height
    maxX=0
    maxY=0
    #pix_arrA = imgA.histogram()
    #im.show()
    for i in range(1,height-1): #using 1 and height -1 since boundary also have pixel values inthe question
        for j in range(1,width-1):
            pixVal = im.getpixel((i,j))
            pixVal2 = im.getpixel((i+1, j+1))
            if pixVal<1:
                # if j< minX:
                #     minX=j
                if i< minY:
                    minY = i
                # if j > maxX:
                #     maxX = j
                if i >  maxY:
                    maxY = i
    #print(minX,minY,maxX,maxY) #need to check X and Y, Y is width and X is height
    imgNew = img.crop((minY,0,maxY,height))
    return imgNew
def using_regiongrowing(ACCURACY):
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
    global ques_img, ans_img
    imgA1 = region_growing(imgA)
    imgB1 = region_growing(imgB)
    imgC1 = region_growing(imgC)
    imgD1 = region_growing(imgD)
    imgE1 = region_growing(imgE)
    imgF1 = region_growing(imgF)
    imgG1 = region_growing(imgG)
    imgH1 = region_growing(imgH)

    imgABody = Image.new('1', imgA.size)
    imgABody = ImageChops.invert(imgABody)
    imgABody.paste(imgA1, (0, 0))
    dstBC = Image.new('1', imgA.size  )
    dstBC = ImageChops.invert(dstBC)
    dstBC.paste(imgB1, (0, 0))
    dstBC.paste(imgC1, (imgB1.width, 0))
    # dstBC.show()
    # imgABody.show()
    diffA_BC = similaritycheck_newThreshold(imgABody,dstBC,ACCURACY)
    #DEF checks
    imgDBody = Image.new('1',imgD.size)
    imgDBody = ImageChops.invert(imgDBody)
    imgDBody.paste(imgD1,(0,0))
    dstEF = Image.new('1',imgD.size)
    dstEF =  ImageChops.invert(dstEF)
    dstEF.paste(imgE1, (0, 0))
    dstEF.paste(imgF1,(imgE1.width, 0))
    # dstEF.show()
    # imgDBody.show()
    diffD_EF = similaritycheck_newThreshold(imgDBody, dstEF, ACCURACY)
    bestdiff = sys.maxsize
    bestans = -1
    found = False
    # print(diffA_BC, diffD_EF)
    if diffA_BC[0] and diffD_EF[0]:
        imgGBody = Image.new('1', imgG.size)
        imgGBody= ImageChops.invert(imgGBody)
        imgGBody.paste(imgG1, (0, 0))
        dstHOpt = Image.new('1', imgG.size)
        dstHOpt =  ImageChops.invert(dstHOpt)
        dstHOpt.paste(imgH1, (0, 0))

        for no in range(len(ans_img)):
            dstHOpt = Image.new('1', imgG.size)
            # dstHOpt.show()
            dstHOpt = ImageChops.invert(dstHOpt)
            # dstHOpt.show()
            dstHOpt.paste(imgH1, (0, 0))
            # dstHOpt.show()
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            imgOpt1 = region_growing(imgOpt)
            dstHOpt.paste(imgOpt1, (imgH1.width, 0))
            diffG_HOpt = similaritycheck_newThreshold(imgGBody, dstHOpt, ACCURACY)
            if diffG_HOpt[0] and diffG_HOpt[1]<bestdiff:
                bestdiff = diffG_HOpt[1]
                bestans = no+1
                found = True
    return found, bestans, bestdiff


def e12Kinds():
    global ques_img, ans_img
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH

    resAB = getCroppedImage2(imgA,imgB)
    resC = getCroppedImage1(imgC)
    # resultdiff.show()
    # img3.show()
    bestdiff = sys.maxsize
    bestans = -1
    found = False
    e12case = similaritycheck_newThreshold(resAB,resC,1) #1 is not useful
    if e12case[0]:
        resGH = getCroppedImage2(imgG,imgH)
        # resGH.show()

        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            resOpt = getCroppedImage1(imgOpt)
            # resOpt.show()
            e12Check = similaritycheck_newThreshold(resGH, resOpt,1)
            if e12Check[0] and e12Check[1]<bestdiff:
                bestdiff = e12Check[1]
                bestans=no+1
                found = True
    resAD = getCroppedImage2(imgA, imgD)
    resG = getCroppedImage1(imgG)
    # resultdiff.show()
    # img3.show()
    # bestdiff = sys.maxsize
    # bestans = -1
    # found = False
    e12case = similaritycheck_newThreshold(resAD, resG, 1)  # 1 is not useful
    if e12case[0]:
        resCF = getCroppedImage2(imgC, imgF)
        # resGH.show()

        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            resOpt = getCroppedImage1(imgOpt)
            # resOpt.show()
            e12Check = similaritycheck_newThreshold(resCF, resOpt, 1)
            if e12Check[0] and e12Check[1] < bestdiff:
                bestdiff = e12Check[1]
                bestans = no + 1
                found = True
    return found, bestans, bestdiff

    # print()
    # resultdiff.show()
def getCroppedImage1(imgc):
    img3 = ImageChops.invert(imgc)
    img3 = imgc.crop(img3.getbbox())
    img3Body = Image.new('1', imgc.size)
    img3Body = ImageChops.invert(img3Body)
    img3Body.paste(img3, (0, 0))
    img3 = img3Body
    return  img3
def getCroppedImage2(imga,imgb):
    img1 = ImageChops.invert(imga)
    img1 = imga.crop(img1.getbbox())
    img1Body = Image.new('1', imga.size)
    img1Body = ImageChops.invert(img1Body)
    img1Body.paste(img1, (0, 0))
    img1 = img1Body
    # img1.show()

    # imgB.show()
    # need to rotate image and crop it properly
    # imgBOrig = imgB.rotate(180)
    # img2.show()
    img2rot = imgb.rotate(180)
    img2rot = ImageChops.invert(img2rot)
    img2 = ImageChops.invert(img2rot)
    # img2.show()
    img2 = img2.crop(img2.getbbox())
    # img2=img2.rotate(180)
    img2Body = Image.new('1', imgb.size)
    img2Body = ImageChops.invert(img2Body)
    img2Body.paste(img2, (0, 0))
    img2 = img2Body
    # img2.show()
    # had to do it again to convert image, cannot figure out why the image is at bottom and not top
    img2b = ImageChops.invert(img2)
    img2b = img2.crop(img2b.getbbox())
    img2bBody = Image.new('1', imgC.size)
    img2bBody = ImageChops.invert(img2bBody)
    img2bBody.paste(img2b, (0, 0))
    img2b = img2bBody
    # img2b.show()



    img1diff2 = ImageChops.difference(img1, img2b)

    # filtering using erosion and dilation
    img_t = img1diff2  # ImageChops.invert(img1diff2)
    img_t = img_t.filter(ImageFilter.MinFilter(3))
    img_t = img_t.filter(ImageFilter.MaxFilter(3))
    img_t = ImageChops.invert(img_t)

    img1diff2 = img_t
    # img1diff2.show()

    imgABdiff = ImageChops.invert(img1diff2)
    # imgABdiff.show()
    imgABdiff = imgABdiff.crop(imgABdiff.getbbox())
    # imgABdiff.show()
    imgABdiff = ImageChops.invert(imgABdiff)
    # imgABdiff.show()
    imgABdiffBody = Image.new('1', imga.size)
    imgABdiffBody = ImageChops.invert(imgABdiffBody)
    imgABdiffBody.paste(imgABdiff, (0, 0))
    # imgABdiffBody.show()

    resultdiff = imgABdiffBody
    return  resultdiff
def check_halfcombined(THRESHOLD):
    #For E-04
    global ques_img, ans_img
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
    [width, height] = imgA.size
    imgA_U = imgA.crop((0, 0, width, height / 2))
    imgA_B = imgA.crop((0, height / 2, width, height))

    imgB_U = imgB.crop((0, 0, width, height / 2))
    imgB_B = imgB.crop((0, height / 2, width, height))

    imgC_U = imgC.crop((0, 0, width, height / 2))
    imgC_B = imgC.crop((0, height / 2, width, height))

    imgD_U = imgD.crop((0, 0, width, height / 2))
    imgD_B = imgD.crop((0, height / 2, width, height))

    imgG_U = imgG.crop((0, 0, width, height / 2))
    imgG_B = imgG.crop((0, height / 2, width, height))

    diff1 = similaritycheck_newThreshold(imgA_U,imgC_U,THRESHOLD)
    diff2 = similaritycheck_newThreshold(imgB_B, imgC_B, THRESHOLD)
    diff3 = similaritycheck_newThreshold(imgA_U, imgG_U, THRESHOLD)
    diff4 = similaritycheck_newThreshold(imgD_B, imgG_B, THRESHOLD)
    bestdiff = sys.maxsize
    bestans = -1
    found = False
    if diff1[0] and diff2[0] and diff3[0] and diff4[0]:
        imgH_B = imgH.crop((0, height / 2, width, height))
        imgF_B = imgF.crop((0, height / 2, width, height))
        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            imgOpt_U = imgOpt.crop((0, 0, width, height / 2))
            imgOpt_B = imgOpt.crop((0, height / 2, width, height))
            diffO1 = similaritycheck_newThreshold(imgG_U, imgOpt_U, THRESHOLD)
            diffO2 = similaritycheck_newThreshold(imgH_B, imgOpt_B, THRESHOLD)
            diffO3 = similaritycheck_newThreshold(imgC_U, imgOpt_U, THRESHOLD)
            diffO4 = similaritycheck_newThreshold(imgF_B, imgOpt_B, THRESHOLD)
            if diffO1[0] and diffO2[0]and diffO3[0]and diffO4[0]:
                diffm = (diffO1[1]+diffO2[1]++diffO3[1]+diffO4[1])/4
                if diffm<bestdiff:
                    bestdiff = diffm
                    bestans = no+1
                    found =True
    return found, bestans, bestdiff
# def summing(THRESHOLD):
#     global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
#     add_AB = ImageChops.logical_and(imgA, imgB)
#     add_AB.show()
#     add_GH = ImageChops.logical_and(imgG, imgH)
#     add_GH.show()
#     #xor_AB = ImageChops.invert(xor_AB).convert("1")
#     same  = similaritycheck_newThreshold(add_AB,imgC, THRESHOLD)
#     bestdiff = sys.maxsize
#     bestans = -1
#     found = False
#     if same[0]:
#         xor_GH = ImageChops.logical_xor(imgG, imgH)
#         xor_GH = ImageChops.invert(xor_GH).convert("1")
#         for no in range(len(ans_img)):
#             imgOpt = ans_img[no]
#             imgOpt = imgOpt.convert("1")
#             #xor_GH.show()
#             same_Opt = similaritycheck_newThreshold(imgOpt, xor_GH, THRESHOLD)
#             same_Opt_mean = same_Opt[1]
#             if same_Opt[0] and same_Opt_mean < bestdiff:
#                 #imgOpt.show()
#                 bestdiff = same_Opt_mean
#                 bestans = no + 1
#                 found = True
#         return found, bestans, bestdiff
#     xor_AD = ImageChops.logical_xor(imgA, imgD)
#     xor_AD = ImageChops.invert(xor_AD).convert("1")
#     same = similaritycheck_newThreshold(xor_AD, imgG, THRESHOLD)
#     bestdiff = sys.maxsize
#     bestans = -1
#     found = False
#     if same[0]:
#         xor_CF = ImageChops.logical_xor(imgC, imgF)
#         xor_CF = ImageChops.invert(xor_CF).convert("1")
#         for no in range(len(ans_img)):
#             imgOpt = ans_img[no]
#             imgOpt = imgOpt.convert("1")
#             same_Opt = similaritycheck_newThreshold(imgOpt, xor_CF, THRESHOLD)
#             same_Opt_mean = same_Opt[1]
#             if same_Opt[0] and same_Opt_mean < bestdiff:
#                 bestdiff = same_Opt_mean
#                 bestans = no + 1
#                 found = True
#         return found, bestans, bestdiff
#     return found, bestans, bestdiff

# def remove_boundary(THRESHOLD): # does not work
#     global ques_img,ans_img
#     global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
#
#     first = [1, 0, 0, 3]
#     second = [0, 3, 1, 0]
#     third = [2, 6, 2, 6]
#     ans_first = [7, 2, 6, 5]
#     ans_second = [6, 5, 7, 2]
#     bestdiff = sys.maxsize
#     bestans = -1
#     found = False
#     for i in range(len(first)):
#         img1 = ques_img[first[i]]
#         img2 = ques_img[second[i]]
#         img3 = ques_img[third[i]]
#         xor_BA = ImageChops.logical_xor(img1, img2)
#         xor_BA = ImageChops.invert(xor_BA).convert("1")
#         xor_BAandC = ImageChops.logical_and(xor_BA, img3)
#         same = similaritycheck_newThreshold(xor_BAandC, imgC, THRESHOLD)
#         bestdiff = sys.maxsize
#         bestans = -1
#         found = False
#         if same[0]:
#             for no in range(len(ans_img)):
#                 imgOpt = ans_img[no]
#                 imgOpt = imgOpt.convert("1")
#                 checkopt = ImageChops.logical_xor(ques_img[ans_first[i]], ques_img[ans_second[i]])
#                 checkopt = ImageChops.invert(checkopt).convert("1")
#                 #checkopt.show()
#                 checkopt = ImageChops.logical_and(checkopt,imgOpt)
#                 diff_Opt = similaritycheck_newThreshold(checkopt, imgOpt, THRESHOLD)
#                 diff_Opt_mean = diff_Opt[1]
#                 if diff_Opt[0] and diff_Opt_mean < bestdiff:
#                     bestdiff = diff_Opt_mean
#                     bestans = no + 1
#                     found = True
#             return found, bestans, bestdiff
#     return found, bestans, bestdiff

def third_is_result(THRESHOLD):
    global ques_img, ans_img
    first = [1,1,0,0,3]
    second = [4,0,3,1,0]
    third = [7,2,6,2,6]
    ans_first = [2,7,2,6,5]
    ans_second = [5,6,5,7,2]
    bestdiff = sys.maxsize
    bestans = -1
    found = False
    for i in range(len(first)):
        img1 = ques_img[first[i]]
        img2 = ques_img[second[i]]
        img3 = ques_img[third[i]]
        bestdiff = sys.maxsize
        leastdiff = sys.maxsize#diff to check for the operations
        bestans = -1
        found = False
        todo = ""
        img12_and = ImageChops.logical_and(img1,img2)
        same_and = similaritycheck_newThreshold ( img12_and, img3, THRESHOLD)
        if same_and[0] and same_and[1] < leastdiff:
            leastdiff = same_and[1]
            todo = "and"

        img12_or = ImageChops.logical_or(img1, img2)
        same_or = similaritycheck_newThreshold(img12_or, img3, THRESHOLD)
        if same_or[0] and same_or[1] < leastdiff:
            leastdiff = same_or[1]
            todo = "or"
        imgsub12  = ImageChops.logical_xor(img1,img2) # this might not be working as expected
        imgsub12 = ImageChops.invert(imgsub12).convert("1")
        # imgsub12.show()
        # img3.show()

        imgsub12b  = ImageChops.difference(img1,img2) # this might not be working as expected
        # imgsub12b.show()
        # imgsub12.show()
        same_sub12 = similaritycheck_newThreshold(imgsub12,img3, THRESHOLD)
        # imgsub12.show()
        # img3.show()
        if same_sub12[0] and same_sub12[1] < leastdiff:
            leastdiff = same_sub12[1]
            todo = "xor"

        # imgadd12 = ImageChops.add(img1, img2)
        same_add12 = similaritycheck_newThreshold(imgsub12, img3, THRESHOLD)
        # imgsub12.show()
        # img3.show()
        if same_add12[0] and same_add12[1] < leastdiff:
            leastdiff = same_add12[1]
            todo = "add"
        if todo == "and":
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                img_sum = ImageChops.logical_and(ques_img[ans_first[i]], ques_img[ans_second[i]])
                diff_Opt = similaritycheck_newThreshold(img_sum,imgOpt,THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt[0] and diff_Opt_mean < bestdiff:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
            return found, bestans, bestdiff
        elif todo == "or":
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                img_sum = ImageChops.logical_or(ques_img[ans_first[i]], ques_img[ans_second[i]])
                diff_Opt = similaritycheck_newThreshold(img_sum, imgOpt, THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt[0] and diff_Opt_mean < bestdiff:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
            return found, bestans, bestdiff
        elif todo == "xor":
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                img_sum = ImageChops.logical_xor(ques_img[ans_first[i]], ques_img[ans_second[i]])
                img_sum = ImageChops.invert(img_sum).convert("1")
                diff_Opt = similaritycheck_newThreshold(img_sum, imgOpt, THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                # img_sum.show()
                # imgOpt.show()
                if diff_Opt[0] and diff_Opt_mean < bestdiff:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
        elif todo == "add":
                for no in range(len(ans_img)):
                    imgOpt = ans_img[no]
                    imgOpt = imgOpt.convert("1")
                    img_sum = ImageChops.add(ques_img[ans_first[i]], ques_img[ans_second[i]])
                    diff_Opt = similaritycheck_newThreshold(img_sum, imgOpt, THRESHOLD)
                    diff_Opt_mean = diff_Opt[1]
                    if diff_Opt[0] and diff_Opt_mean < bestdiff:
                        bestdiff = diff_Opt_mean
                        bestans = no + 1
                        found = True
                return found, bestans, bestdiff





    return found, bestans, bestdiff

def when_addedrows_same(THRESHOLD):
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
    global ans_img
    case=""
    pix_arrA= imgA.histogram()[0]
    pix_arrB= imgB.histogram()[0]
    pix_arrC= imgC.histogram()[0]
    pix_arrD= imgD.histogram()[0]
    pix_arrE= imgE.histogram()[0]
    pix_arrF= imgF.histogram()[0]
    pix_arrG = imgG.histogram()[0]
    pix_arrH = imgH.histogram()[0]

    bestdiff = sys.maxsize
    bestans = -1
    found = False
    threshold=sys.maxsize
    checkCenterResult = check_center_same(THRESHOLD)
    if checkCenterResult[0] and checkCenterResult[2] < bestdiff:
        bestdiff = checkCenterResult[2]
        bestans = checkCenterResult[1]
        found = checkCenterResult[0]
        threshold = bestdiff

    sum_pix_row1 = pix_arrA+pix_arrB+pix_arrC
    sum_pix_row2 = pix_arrD+pix_arrE+pix_arrF

    if sum_pix_row1>sum_pix_row2:
        pix_diff = (sum_pix_row1/sum_pix_row2)
    else:
        pix_diff = (sum_pix_row2/sum_pix_row1)
    pix_cnt_same = False

    if pix_diff>=1 and pix_diff<1.5:
        pix_cnt_same= True
    if pix_cnt_same: # it is either and or or operations for rows and not cols
        imgABC_add = ImageChops.logical_and(imgA, ImageChops.logical_and(imgB, imgC))
        imgDEF_add = ImageChops.logical_and(imgD, ImageChops.logical_and(imgE, imgF))
        # imgABC_add.show()
        # imgDEF_add.show()

        same = similaritycheck_newThreshold(imgABC_add,imgDEF_add, THRESHOLD)

        best_ratio = sys.maxsize
        if same[0] and same[1]<= threshold and pix_cnt_same:
            threshold = same[1]
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                pix_cnt_Opt = pix_arrG+pix_arrH+imgOpt.histogram()[0]
                if sum_pix_row1> pix_cnt_Opt:
                    pix_ratio = sum_pix_row1/pix_cnt_Opt
                else:
                    pix_ratio = pix_cnt_Opt/sum_pix_row1
                pix_ratio_test = False
                if pix_ratio >= 1 and pix_ratio < 1.5:
                    pix_ratio_test = True
                imgGHOpt= ImageChops.logical_and(imgG,ImageChops.logical_and(imgH,imgOpt))
                diff_Opt = similaritycheck_newThreshold(imgDEF_add,imgGHOpt,THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt[0] and diff_Opt_mean < bestdiff and pix_ratio_test and pix_ratio < best_ratio:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
                    best_ratio = pix_ratio
                    case="andof ABC andofDEF same"
            #return found, bestans, bestdiff
    #removing or as it is causing a lot of issues
        # trying if oring images in rows have same results
        imgABC_or = ImageChops.logical_or(imgA, ImageChops.logical_or(imgB, imgC))
        imgDEF_or = ImageChops.logical_or(imgD, ImageChops.logical_or(imgE, imgF))
        pixcnt_ABC = imgABC_or.histogram()[0]
        pixcnt_DEF = imgDEF_or.histogram()[0]
        # imgABC_or.show()
        # imgDEF_or.show()


        # imgABC_or.show()
        # imgDEF_or.show()
        same = similaritycheck_newThreshold(imgABC_or, imgDEF_or, THRESHOLD)
        # mean_diff = same[1]
        best_ratio = sys.maxsize # added best ratio since we want the closest answer
        if same[0] and same[1] <= threshold and pixcnt_ABC >10 and pixcnt_DEF >10:
            threshold = same[1]
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                pix_cnt_Opt = pix_arrG + pix_arrH + imgOpt.histogram()[0]
                if sum_pix_row1 > pix_cnt_Opt:
                    pix_ratio = sum_pix_row1 / pix_cnt_Opt
                else:
                    pix_ratio = pix_cnt_Opt / sum_pix_row1
                pix_ratio_test = False
                if 1 <= pix_ratio < 1.5:
                    pix_ratio_test = True
                imgGHOpt = ImageChops.logical_or(imgG, ImageChops.logical_or(imgH, imgOpt))
                diff_Opt = similaritycheck_newThreshold(imgDEF_or, imgGHOpt,THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt_mean < bestdiff and pix_ratio_test and pix_ratio<best_ratio:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
                    best_ratio = pix_ratio
            #return found, bestans, bestdiff
   ############################################
    #checking for the cols
    sum_pix_1 = pix_arrA+pix_arrD+pix_arrG
    sum_pix_2 = pix_arrB+pix_arrE+pix_arrH

    if sum_pix_1>sum_pix_2:
        pix_diff = (sum_pix_1/sum_pix_2)
    else:
        pix_diff = (sum_pix_2/sum_pix_1)
    pix_cnt_same = False
    if pix_diff>=1 and pix_diff<1.5:
        pix_cnt_same= True
    if pix_cnt_same: # case where columns either and or or operations
        imgADG_add = ImageChops.logical_and(imgA, ImageChops.logical_and(imgD, imgG))
        imgBEH_add = ImageChops.logical_and(imgB, ImageChops.logical_and(imgE, imgH))

        same = similaritycheck_newThreshold(imgADG_add, imgBEH_add, THRESHOLD)
        mean_diff = same[1]
        best_ratio = sys.maxsize  # added best ratio since we want the closest answer
        if same[0] and mean_diff <= threshold:
            threshold = mean_diff
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                pix_cnt_Opt = pix_arrC + pix_arrF + imgOpt.histogram()[0]
                if sum_pix_1 > pix_cnt_Opt:
                    pix_ratio = sum_pix_1 / pix_cnt_Opt
                else:
                    pix_ratio = pix_cnt_Opt / sum_pix_1
                pix_ratio_test = False
                if pix_ratio >= 1 and pix_ratio < 1.5:
                    pix_ratio_test = True
                imgCFOpt = ImageChops.logical_and(imgC, ImageChops.logical_and(imgF, imgOpt))
                diff_Opt = similaritycheck_newThreshold(imgBEH_add, imgCFOpt,THRESHOLD)
                # imgBEH_add.show()
                # imgCFOpt.show()
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt_mean < bestdiff and pix_ratio_test and pix_ratio < best_ratio:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
                    best_ratio = pix_ratio
            #return found, bestans, bestdiff
        #or or columns testing ,removing or as it is causing a lot of issues
        imgADG_or = ImageChops.logical_or(imgA, ImageChops.logical_or(imgD, imgG))
        imgBEH_or = ImageChops.logical_or(imgB, ImageChops.logical_or(imgE, imgH))
        # imgADG_or.show()
        # imgBEH_or.show()
        pixcnt_ADG = imgADG_or.histogram()[0]
        pixcnt_BEH = imgBEH_or.histogram()[0]
        # imgADG_or.show()
        # imgBEH_or.show()

        sim = similaritycheck_newThreshold(imgADG_or,imgBEH_or, THRESHOLD)
        mean_diff = sim[1]
        best_ratio = sys.maxsize  # added best ratio since we want the closest answer
        if sim[0] and mean_diff<threshold and pixcnt_ADG>5 and pixcnt_BEH > 5:
            threshold = mean_diff
            for no in range(len(ans_img)):
                imgOpt = ans_img[no]
                imgOpt = imgOpt.convert("1")
                pix_cnt_Opt = pix_arrC + pix_arrF + imgOpt.histogram()[0]
                if sum_pix_1 > pix_cnt_Opt:
                    pix_ratio = sum_pix_1 / pix_cnt_Opt
                else:
                    pix_ratio = pix_cnt_Opt / sum_pix_1
                pix_ratio_test = False
                if pix_ratio >= 1 and pix_ratio < 1.5:
                    pix_ratio_test = True
                imgCFOpt = ImageChops.logical_or(imgC, ImageChops.logical_or(imgF, imgOpt))
                diff_Opt = similaritycheck_newThreshold(imgBEH_or, imgCFOpt,THRESHOLD)
                diff_Opt_mean = diff_Opt[1]
                if diff_Opt[0] and diff_Opt_mean < bestdiff and pix_ratio_test and pix_ratio < best_ratio:
                    bestdiff = diff_Opt_mean
                    bestans = no + 1
                    found = True
                    best_ratio = pix_ratio
            #return found, bestans, bestdiff
    ########################################
    #checking for D-04, also works for D-05
    row1 = ImageChops.logical_xor(imgC,ImageChops.logical_and(imgA, imgB))
    row1 = ImageChops.invert(row1)
    # row1.show()
    row2 = ImageChops.logical_xor(imgF, ImageChops.logical_and(imgD, imgE))
    row2 = ImageChops.invert(row2)
    # row2.show()
    diff = similaritycheck_newThreshold(row1,row2,THRESHOLD)
    if diff[0] and diff[1]<threshold:
        threshold=diff[1]
        found   = False
        bestans = 0
        bestdiff = sys.maxsize
        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            row3 = ImageChops.logical_xor(imgOpt, ImageChops.logical_and(imgG, imgH))
            row3 = ImageChops.invert(row3)
            diffOpt = similaritycheck_newThreshold(row1,row3,THRESHOLD)
            if diffOpt[0] and diffOpt[1]<bestdiff:
                found =True
                bestdiff = diffOpt[1]
                bestans = no+1
        #return found, bestans, bestdiff
    #Testing for D-05
    # col1 = ImageChops.logical_xor(imgG, ImageChops.logical_and(imgA, imgD))
    # col1 = ImageChops.invert(col1)
    # col1.show()
    # col2 = ImageChops.logical_xor(imgH, ImageChops.logical_and(imgH, imgE))
    # col2 = ImageChops.invert(col2)
    # col2.show()
    thirisresult = third_is_result(THRESHOLD)
    if thirisresult[0] and thirisresult[2] < bestdiff:
        bestdiff = thirisresult[1]
        bestans = thirisresult[1]
        found = True
    halfcombinedresult = check_halfcombined(THRESHOLD)
    if halfcombinedresult[0] and halfcombinedresult[2]<bestdiff:
        bestdiff = halfcombinedresult[2]
        bestans = halfcombinedresult[1]

    return found, bestans, bestdiff
def find_startcoord(img,irangeMin, irangeMax, jrangeMin, jrangeMax):
    for i in range(irangeMin, irangeMax):
        for j in range(jrangeMin, jrangeMax):
            pix = img.getpixel((j, i))
            if pix > 1:
                return i


def find_endcoord(img, irangeMin, irangeMax, jrangeMin, jrangeMax):
    for i in range(irangeMin, irangeMax,-1):
        for j in range(jrangeMin, jrangeMax):
            pix = img.getpixel((i,j))
            if pix > 1:
                return i
def get_imageonly(img):
    [w,h]=img.size
    # wstart=w
    # wend=0
    # hstart=h
    # hend=0
    # his = img.histogram()
    # img.show()
    hstart = find_startcoord(img,2,h,2,w)
    # for i in range(2,h):
    #     for j in range(2,w):
    #         pix = img.getpixel((i,j))
    #         if pix>1:
    #             hstart = i
    #             break
    #     else:
    #         continue
    hend = find_endcoord(img,h-1,2,2,w)

    # for i in range(h-1,2,-1):
    #     for j in range(2,w):
    #         pix = img.getpixel((j,i))
    #         if pix>1:
    #             hend = i
    #             break
    #         else:
    #             continue
    wstart = find_startcoord(img,2,w,2,h )
    # for i in range(2,w):
    #     for j in range(2,h):
    #         pix = img.getpixel((i, j))
    #         if pix > 1:
    #             wstart = i
    #             break
    #         else:
    #             continue
    wend = find_endcoord(img,w-1,2,2,h)

    # for i in range(w-1,2,-1):
    #     for j in range(2,h):
    #         pix = img.getpixel((j,i))
    #         if pix>1:
    #             wend = i
    #             break
    #         else:
    #             continue
    # print(wstart,hstart,wend,hend)

    imgBody = Image.new('1', img.size)
    imgBody = ImageChops.invert(imgBody)
    # imgBody.show()
    img1 = img.crop((wstart,hstart,wend,hend))
    # img1.show()
    img1hist = img1.histogram()
    imgBody.paste(img1, (0, 0))
    return imgBody
    # imgBody.show()

# def similaritycheck_newThreshold(img1,img2, THRESHOLD):
#
#     img1_p = img1.histogram()[0]
#     img2_p = img2.histogram()[0]
#     img1_pw = img1.histogram()[255]
#     img2_pw = img1.histogram()[255]
#     if img1_p ==0:
#         img1_p=1
#     if img1_pw == 0:
#         img1_pw = 1
#     if img2_p == 0:
#         img2_p = 1
#     if img2_pw == 0:
#         img2_pw = 1
#
#     #w = img2.histogram()[254]
#     # img1New = get_imageonly(img1)
#     #images could be completey dark or white
#     bwratio1= 0
#     moreW= False
#     moreB = False
#     if img1_p > img2_pw:
#         bwratio1 = img1_p/img1_pw
#         moreB = True
#     else:
#         bwratio1 = img2_pw / img1_p
#         moreW = True
#     bwratio2 = 0
#     if img2_p > img2_pw:
#         bwratio2 = img2_p / img2_pw
#     else:
#         bwratio2 = img2_pw / img1_p
#     wToB = img2_pw/img2_p
#     [w,h] = img1.size
#     if img1_p==0:
#         img1_p=1
#     if img2_p==0:
#         img2_p = 1
#     ratio1 = w*h/img1_p
#     ratio2 = w * h / img2_p
#     pix_diff = abs(img1_p-img2_p)
#     diff_mean = sys.maxsize
#
#
#     # if pix_diff > 250:
#     #     same = False
#     #     return same, diff_mean
#     [w,h] = img1.size
#     img1_1 = img1.crop((0,0,w/2,h/2))
#     img1_2 = img1.crop((w/2,0,w,h/2))
#     img1_3 = img1.crop((0, h/2, w / 2, h ))
#     img1_4 = img1.crop((w / 2, h/2, w, h ))
#     img2_1 = img2.crop((0,0,w/2,h/2))
#     img2_2 = img2.crop((w/2,0,w,h/2))
#     img2_3 = img2.crop((0, h/2, w / 2, h ))
#     img2_4 = img2.crop((w / 2, h/2, w, h ))
#     # if (ratio1 > 0.95 and ratio1 < 1.5) or(ratio2 > 0.95 and ratio2 < 1.5) : #case where both images are completely black
#     #     same = False
#     #     return same, diff_mean
#     THRESHOLD = (ratio1+ratio2)/2
#
#     #THRESHOLD = (bwratio1+bwratio2)/2
#     diff  = np.logical_xor(img1,img2)
#     diff1 = np.logical_xor(img1_1, img2_1)
#     diff2 = np.logical_xor(img1_2, img2_2)
#     diff3 = np.logical_xor(img1_3, img2_3)
#     diff4 = np.logical_xor(img1_4, img2_4)
#
#
#     diff_mean  = diff.mean() * 100
#     diff_mean1 = diff1.mean() * 100
#     diff_mean2 = diff2.mean() * 100
#     diff_mean3 = diff3.mean() * 100
#     diff_mean4 = diff4.mean() * 100
#
#     Threshold2 = THRESHOLD
#     k =1.9
#     if THRESHOLD >4:
#          Threshold2 = k*THRESHOLD / 4
#     #Threshold2 =  THRESHOLD
#     partsCnt=0
#     if diff_mean1 < Threshold2:
#         partsCnt+=1
#     if diff_mean2 < Threshold2:
#         partsCnt += 1
#     if diff_mean3 < Threshold2:
#         partsCnt += 1
#     if diff_mean4 < Threshold2:
#         partsCnt += 1
#
#     same = False
#     if img1_p<300 or img2_p<300:
#         THRESHOLD = 0.5
#     if img1_p < 200 or img2_p < 200:
#             THRESHOLD = 0.3
#     if img1_p < 100 or img2_p < 100:
#             THRESHOLD = 0.1
#     # if moreB:
#     #     THRESHOLD = wToB * 4
#     if diff_mean < THRESHOLD and partsCnt>=3:# diff_mean1 < Threshold2 and diff_mean2 < Threshold2 and diff_mean3 < Threshold2 and diff_mean4 < Threshold2:
#         same = True
#     return same, diff_mean
def similaritycheck_newThreshold(img1,img2, THRESHOLD):
    img1orig = img1
    img2orig = img2

    img1 = img1.crop(img1.getbbox())
    img1Body = Image.new('1', img1orig.size)
    img1Body = ImageChops.invert(img1Body)
    # imgBody.show()
    img1Body.paste(img1,(0,0))
    img2 = img2.crop(img1.getbbox())
    img2Body = Image.new('1', img1orig.size)
    img2Body = ImageChops.invert(img2Body)
    # imgBody.show()
    img2Body.paste(img2, (0, 0))
    img1 = img1Body
    img2 = img2Body


    #processing of image: eroding and dilating
    img1 = ImageChops.invert(img1)
    img1 = img1.filter(ImageFilter.MinFilter(3))
    img1 = img1.filter(ImageFilter.MaxFilter(3))
    img1 = ImageChops.invert(img1)

    img2 = ImageChops.invert(img2)
    img2 = img2.filter(ImageFilter.MinFilter(3))
    img2 = img2.filter(ImageFilter.MaxFilter(3))
    img2 = ImageChops.invert(img2)

    [w,h] = img1.size
    img1_p = img1.histogram()[0]
    img2_p = img2.histogram()[0]
    img1_pw = img1.histogram()[255]
    img2_pw = img2.histogram()[255]

    # if (img1_pw+img2_pw)/2 >

    if img1_p<5 and img2_p<5:
        return False, sys.maxsize
    if img1_pw<5 and img2_p<5:
        return False, sys.maxsize
    # img1.show()
    # img2.show()
    totalsize = w*h
    pctofPixels = 0.05*(w*h)
    newThreshold = 0.02*img1_pw
    avgWP = (img1_pw+img2_pw)/2
    avgBP = (img1_p+img2_p)/2
    avgP=avgWP
    if avgBP>avgWP:
        avgP = avgBP

    if img1_p ==0:
        img1_p=1
    if img1_pw == 0:
        img1_pw = 1
    if img2_p == 0:
        img2_p = 1
    if img2_pw == 0:
        img2_pw = 1

    #images could be completey dark or white
    bwratio1= 0
    moreW= False
    moreB = False
    if img1_p > img2_pw:
        bwratio1 = img1_p/img1_pw
        moreB = True
    else:
        bwratio1 = img2_pw / img1_p
        moreW = True
    bwratio2 = 0
    if img2_p > img2_pw:
        bwratio2 = img2_p / img2_pw
    else:
        bwratio2 = img2_pw / img1_p
    wToB = img2_pw/img2_p

    if img1_p==0:
        img1_p=1
    if img2_p==0:
        img2_p = 1
    ratio1 = w*h/img1_p
    ratio2 = w * h / img2_p
    pix_diff = abs(img1_p-img2_p)
    diff_mean = sys.maxsize

    [w,h] = img1.size
    img1_1 = img1.crop((0,0,w/2,h/2))
    img1_2 = img1.crop((w/2,0,w,h/2))
    img1_3 = img1.crop((0, h/2, w / 2, h ))
    img1_4 = img1.crop((w / 2, h/2, w, h ))
    img2_1 = img2.crop((0,0,w/2,h/2))
    img2_2 = img2.crop((w/2,0,w,h/2))
    img2_3 = img2.crop((0, h/2, w / 2, h ))
    img2_4 = img2.crop((w / 2, h/2, w, h ))
    # if (ratio1 > 0.95 and ratio1 < 1.5) or(ratio2 > 0.95 and ratio2 < 1.5) : #case where both images are completely black
    #     same = False
    #     return same, diff_mean
    # THRESHOLD = (ratio1+ratio2)/2
    #
    # THRESHOLD = (bwratio1+bwratio2)/2

    THRESHOLD = avgWP
    diff  = np.logical_xor(img1,img2)
    diff1 = np.logical_xor(img1_1, img2_1)
    diff2 = np.logical_xor(img1_2, img2_2)
    diff3 = np.logical_xor(img1_3, img2_3)
    diff4 = np.logical_xor(img1_4, img2_4)


    diff_mean  = diff.mean() * 100
    diff_mean1 = diff1.mean() * 100
    diff_mean2 = diff2.mean() * 100
    diff_mean3 = diff3.mean() * 100
    diff_mean4 = diff4.mean() * 100
    Threshold2 = 1.2*THRESHOLD #changing to make it work for E-04
    k =1.9
    if THRESHOLD >2:
         Threshold2 = k*THRESHOLD / 4
    #Threshold2 =  THRESHOLD
    newT = 5#6
    THRESHOLD = avgP/(w*h)*newT
    Threshold2 = THRESHOLD
    partsCnt=0
    if diff_mean1 < Threshold2:
        partsCnt+=1
    if diff_mean2 < Threshold2:
        partsCnt += 1
    if diff_mean3 < Threshold2:
        partsCnt += 1
    if diff_mean4 < Threshold2:
        partsCnt += 1

    same = False
    if img1_p<300 or img2_p<300:
        THRESHOLD = 0.5
    if img1_p < 200 or img2_p < 200:
            THRESHOLD = 0.3
    if img1_p < 100 or img2_p < 100:
            THRESHOLD = 0.1
    # if moreB:
    #     THRESHOLD = wToB * 4
    if diff_mean < THRESHOLD and partsCnt>=4:# diff_mean1 < Threshold2 and diff_mean2 < Threshold2 and diff_mean3 < Threshold2 and diff_mean4 < Threshold2:
        # img1.show()
        # img1 = ImageChops.invert(img1)
        # img1=img1.filter(ImageFilter.MinFilter(3))
        # img1 = img1.filter(ImageFilter.MaxFilter(3))
        # img1 = ImageChops.invert(img1)
        # img1.show()
        # img2.show()
        same = True
    return same, diff_mean

def similaritycheck_newThreshold_bkup(img1, img2, THRESHOLD):
    # img1 = get_imageonly(img1)
    # img1 = ImageChops.invert(img1)
    # img2 = get_imageonly(img2)
    # img2 = ImageChops.invert(img2)
    # img1.show()
    # img2.show()
    # img1 = img1.filter(ImageFilter.MinFilter(5))
    # img1 = img1.filter(ImageFilter.MaxFilter(3))
    # # img1.show()
    # img2 = img2.filter(ImageFilter.MinFilter(5))
    # img2 = img2.filter(ImageFilter.MaxFilter(3))

    img1_p = img1.histogram()[0]
    img2_p = img2.histogram()[0]
    img1_pw = img1.histogram()[255]
    img2_pw = img1.histogram()[255]
    if img1_p == 0:
        img1_p = 1
    if img1_pw == 0:
        img1_pw = 1
    if img2_p == 0:
        img2_p = 1
    if img2_pw == 0:
        img2_pw = 1

    # w = img2.histogram()[254]
    # img1New = get_imageonly(img1)
    # images could be completey dark or white
    bwratio1 = 0
    moreW = False
    moreB = False
    if img1_p > img2_pw:
        bwratio1 = img1_p / img1_pw
        moreB = True
    else:
        bwratio1 = img2_pw / img1_p
        moreW = True
    bwratio2 = 0
    if img2_p > img2_pw:
        bwratio2 = img2_p / img2_pw
    else:
        bwratio2 = img2_pw / img1_p
    wToB = img2_pw / img2_p
    [w, h] = img1.size
    if img1_p == 0:
        img1_p = 1
    if img2_p == 0:
        img2_p = 1
    ratio1 = w * h / img1_p
    ratio2 = w * h / img2_p
    pix_diff = abs(img1_p - img2_p)
    diff_mean = sys.maxsize

    # if pix_diff > 250:
    #     same = False
    #     return same, diff_mean
    [w, h] = img1.size
    img1_1 = img1.crop((0, 0, w / 2, h / 2))
    img1_2 = img1.crop((w / 2, 0, w, h / 2))
    img1_3 = img1.crop((0, h / 2, w / 2, h))
    img1_4 = img1.crop((w / 2, h / 2, w, h))
    img2_1 = img2.crop((0, 0, w / 2, h / 2))
    img2_2 = img2.crop((w / 2, 0, w, h / 2))
    img2_3 = img2.crop((0, h / 2, w / 2, h))
    img2_4 = img2.crop((w / 2, h / 2, w, h))
    # if (ratio1 > 0.95 and ratio1 < 1.5) or(ratio2 > 0.95 and ratio2 < 1.5) : #case where both images are completely black
    #     same = False
    #     return same, diff_mean
    THRESHOLD = (ratio1 + ratio2) / 2

    # THRESHOLD = (bwratio1+bwratio2)/2
    diff = np.logical_xor(img1, img2)
    diff1 = np.logical_xor(img1_1, img2_1)
    diff2 = np.logical_xor(img1_2, img2_2)
    diff3 = np.logical_xor(img1_3, img2_3)
    diff4 = np.logical_xor(img1_4, img2_4)

    diff_mean = diff.mean() * 100
    diff_mean1 = diff1.mean() * 100
    diff_mean2 = diff2.mean() * 100
    diff_mean3 = diff3.mean() * 100
    diff_mean4 = diff4.mean() * 100

    Threshold2 = THRESHOLD
    k = 1.9
    if THRESHOLD > 4:
        Threshold2 = k * THRESHOLD / 4
    # Threshold2 =  THRESHOLD
    partsCnt = 0
    if diff_mean1 < Threshold2:
        partsCnt += 1
    if diff_mean2 < Threshold2:
        partsCnt += 1
    if diff_mean3 < Threshold2:
        partsCnt += 1
    if diff_mean4 < Threshold2:
        partsCnt += 1

    same = False
    if img1_p < 300 or img2_p < 300:
        THRESHOLD = 0.5
    if img1_p < 200 or img2_p < 200:
        THRESHOLD = 0.3
    if img1_p < 100 or img2_p < 100:
        THRESHOLD = 0.1
    # if moreB:
    #     THRESHOLD = wToB * 4
    if diff_mean < THRESHOLD and partsCnt >= 3:  # diff_mean1 < Threshold2 and diff_mean2 < Threshold2 and diff_mean3 < Threshold2 and diff_mean4 < Threshold2:
        same = True
    return same, diff_mean

def check_center_same(THRESHOLD):
    global imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH
    global ans_img
    threshold = 1.8
    img1 =imgA
    img2=imgB
    img3=imgC
    img4=imgD
    img5=imgE
    img6=imgF
    img7=imgG
    img8=imgH
    #combine xor of A B C
    ################################################33
    #testing for horizontal
    n=7
    #img1= ImageChops.invert(imgA)
    # img1.show()
    # img1 = imgA.filter(ImageFilter.MinFilter(n))
    #
    im1hist = img1.histogram()
    #img1 = ImageChops.invert(img1)
    # img1.show()
    # img2 = imgB.filter(ImageFilter.MinFilter(n))
    # img2.show()
    # img3 = imgC.filter(ImageFilter.MinFilter(n))
    or_AB=ImageChops.logical_or(img1,img2)
    # img3.show()
    xor_ABC = ImageChops.logical_xor(or_AB,img3)
    # xor_ABC.show()
    # combine xor of D E F
    # img4 = imgD.filter(ImageFilter.MinFilter(n))
    # img5 = imgE.filter(ImageFilter.MinFilter(n))
    # img6 = imgF.filter(ImageFilter.MinFilter(n))
    or_DE = ImageChops.logical_or(img4, img5)
    xor_DEF = ImageChops.logical_xor(or_DE, img6)
    # xor_DEF.show()
    diff_xors12row = similaritycheck_newThreshold(xor_ABC,xor_DEF,THRESHOLD)#np.logical_xor(xor_ABC,xor_DEF)
    bestdiff=sys.maxsize
    bestans=-1
    foundans=False
    # img7 = imgG.filter(ImageFilter.MinFilter(n))
    # img8 = imgH.filter(ImageFilter.MinFilter(n))
    if diff_xors12row[0] and diff_xors12row[1] < threshold: #meandiff12row < threshold:
        threshold = diff_xors12row[1]
        or_GH = ImageChops.logical_or(img7,img8)
        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            # imgOpt = imgOpt.filter(ImageFilter.MinFilter(n))
            xor_GHOpt = ImageChops.logical_xor(or_GH,imgOpt)
            # xor_GHOpt.show()
            diff23 = similaritycheck_newThreshold(xor_DEF,xor_GHOpt,THRESHOLD)
            if diff23[0] and diff23[1]<bestdiff:
                bestdiff = diff23[1]
                bestans = no+1
                foundans = True
    ################################################33
    # testing for vertical
    xor_AD = ImageChops.logical_or(imgA, imgD)
    xor_ADG = ImageChops.logical_xor(xor_AD, imgG)
    xor_BE = ImageChops.logical_or(imgB, imgE)
    xor_BEH = ImageChops.logical_xor(xor_BE, imgH)
    diff_xors12col = similaritycheck_newThreshold(xor_ADG, xor_BEH,THRESHOLD)

    if diff_xors12col[0] and diff_xors12col[1] < threshold:
        threshold = diff_xors12col[1]
        or_CF = ImageChops.logical_or(imgC, imgF)
        for no in range(len(ans_img)):
            imgOpt = ans_img[no]
            imgOpt = imgOpt.convert("1")
            xor_CFOpt = ImageChops.logical_xor(or_CF, imgOpt)
            diff23 = similaritycheck_newThreshold(xor_BEH,xor_CFOpt,THRESHOLD)
            if diff23[0] and diff23[1] < bestdiff:
                bestdiff = diff23[1]
                bestans = no + 1
                foundans = True
    ###################################################
    #testing for D06
    or_ADG = ImageChops.logical_and(imgA, ImageChops.logical_and(imgD,imgG))
    or_BEH = ImageChops.logical_and(imgB,ImageChops.logical_and(imgE,imgH))
    # or_ADG.show()
    # or_BEH.show()
    pix_ADG  = imgA.histogram()[0]+imgD.histogram()[0]+imgG.histogram()[0]
    pix_BEH  = imgB.histogram()[0]+imgE.histogram()[0]+imgH.histogram()[0]
    pix_diff= abs(pix_ADG-pix_BEH)
    pix_thres = 30
    #threshold=2.5
    diff_ADG_BEH = similaritycheck_newThreshold(or_ADG,or_BEH,THRESHOLD)
    #mean_diff_ADG_BEH = diff_ADG_BEH.mean()*100
    if diff_ADG_BEH[0] and diff_ADG_BEH[1] < threshold and pix_diff<pix_thres:
        threshold = diff_ADG_BEH[1]
        for nos in range(len(ans_img)):
            opt = ans_img[nos]
            or_CFopt = ImageChops.logical_and(opt,ImageChops.logical_and(imgC,imgF))
            pix_CFopt = imgC.histogram()[0] + imgF.histogram()[0] + opt.histogram()[0]
            pix_diff_opt = abs(pix_BEH-pix_CFopt)
            diff_BEH_CFopt = similaritycheck_newThreshold(or_BEH,or_CFopt,THRESHOLD)
            if diff_BEH_CFopt[0] and diff_BEH_CFopt[1] < bestdiff and pix_diff_opt<pix_thres:
                #or_CFopt.show()
                bestdiff = diff_BEH_CFopt[1]
                bestans = nos+1
                foundans = True
                threshold = bestdiff
    #testing for D-09, has been done already
    # sum_ABC = ImageChops.logical_and(imgA, ImageChops.logical_and(imgB, imgC))
    # sum_DEF = ImageChops.logical_and(imgD, ImageChops.logical_and(imgE, imgF))
    # # sum_ABC.show()
    # # sum_DEF.show()
    # pix_ABC = imgA.histogram()[0] + imgB.histogram()[0] + imgC.histogram()[0]
    # pix_DEF = imgD.histogram()[0] + imgE.histogram()[0] + imgF.histogram()[0]
    # pix_diff = abs(pix_ABC - pix_DEF)
    # pix_thres = 60
    # # threshold=2.5
    # diff_ABC_DEF = similaritycheck_newThreshold(sum_ABC, sum_DEF, THRESHOLD)
    # # mean_diff_ADG_BEH = diff_ADG_BEH.mean()*100
    # if diff_ABC_DEF[0] and diff_ABC_DEF[1] < threshold and pix_diff < pix_thres:
    #     threshold = diff_ABC_DEF[1]
    #     for nos in range(len(ans_img)):
    #         opt = ans_img[nos]
    #         and_GHopt = ImageChops.logical_and(opt, ImageChops.logical_and(imgG, imgH))
    #         pix_GHopt = imgG.histogram()[0] + imgH.histogram()[0] + opt.histogram()[0]
    #         pix_diff_opt = abs(pix_DEF - pix_GHopt)
    #         diff_ABC_GHopt = similaritycheck_newThreshold(sum_ABC, and_GHopt, THRESHOLD)
    #         # and_GHopt.show()
    #         # sum_ABC.show()
    #         diff_DEF_GHopt = similaritycheck_newThreshold(sum_DEF, and_GHopt, THRESHOLD)
    #         if diff_ABC_GHopt[0] and diff_ABC_GHopt[1] < bestdiff and pix_diff_opt< pix_thres:
    #             # or_CFopt.show()
    #             bestdiff = diff_ABC_GHopt[1]
    #             bestans = nos + 1
    #             foundans = True

    return foundans, bestans, bestdiff






def check_similarity(img1_q, img2_q, img3_q, test_img, ACCURACY_THRES):
    best_ans=-1
    best_diff=sys.maxsize
    global ques_img, hist_img, ans_img
    img1 = img1_q.convert('1')
    #img1.show()
    img2 = img2_q.convert('1')
    #img2.show()
    img3 = img3_q.convert('1')
   # img3.show()
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

def diagonalSame(ACCURACY_THRES):
    best_ans = -1
    best_diff = sys.maxsize
    similar = False
    global ques_img, ans_img
    imgA = ques_img[0]
    imgE = ques_img[4]
    imgB = ques_img[1]
    imgF = ques_img[5]
    diff_A_E = np.logical_xor(imgA,imgE)
    m_diffAE = diff_A_E.mean()*100
    diff_B_F=np.logical_xor(imgB,imgF)
    m_diffBF = diff_B_F.mean()*100
    if m_diffAE < ACCURACY_THRES and m_diffBF < ACCURACY_THRES:
        diff_temp = find_similar(ans_img, imgE)
        if diff_temp[1] < best_diff:  # find the smallest difference from the options
            best_diff = diff_temp[1]
            best_ans = diff_temp[0]
            similar = True
    return similar, best_ans, best_diff


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
                    if abs(
                            ratio_ans_F - ratio_ans_H) < thres:  # if the ratio of option to F and H is the same then it could be the ans
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
    diff_mean = diff.mean()*100
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

