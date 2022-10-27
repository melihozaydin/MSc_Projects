def im_hist(im):
    hist = np.zeros(256, dtype=np.int)
    width = im.shape[0]
    height = im.shape[1]
    for i in range(width):
        for j in range(height):
            a = im[i,j]
            hist[a] += 1
    return hist

def cum_sum(hist):
    cum_hist = np.copy(hist)
    for i in range(1,len(hist)):
    cum_hist[i] = hist[i] + cum_hist[i-1]
    return cum_hist

def hist_match(img, img_ref):
    width = img.shape[0]
    height = img.shape[1]
    hist_img = im_hist(img)
    hist_ref = im_hist(img_ref)
    cum_img = cum_sum(hist_img)
    cum_ref = cum_sum(hist_ref)
    cum_img = cum_img/max(cum_img)
    cum_ref = cum_ref/max(cum_ref)
    n = 255
    new_values = np.zeros((n))
    img_new = np.copy(img)
    for i in range(n):
        j = n - 1
        while True:
            new_values[i] = j
            j = j - 1
            if j < 0 or cum_img[i] > cum_ref[j]:
                break
    for i in np.arange(width):
        for j in np.arange(height):
            a = img_new[i,j]
            b = new_values[a]
            img_new[i,j] = b
    return img_new