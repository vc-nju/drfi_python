NUMBER_THRESHOLD = 256
import numpy as np
import cv2
EPS = 0.00000000000000001

def evaluate_(resS, gt, precision, recall, tpr, fpr):
    gtFM =  gt
    gtFM = cv2.compare(gtFM, 128, cv2.CMP_GT)
    gtBM = cv2.bitwise_not(gtFM)
    gtF = np.sum(gtFM)
    gtB = resS.shape[0] * resS.shape[1] * 255 - gtF
    mae = 0.
    for i in range(NUMBER_THRESHOLD):
        resM = np.zeros(resS.shape)
        tpM = np.zeros(resS.shape)
        fpM = np.zeros(resS.shape)
        resM = cv2.compare(resS, i, cv2.CMP_GT)
        tpM = cv2.bitwise_and(resM, gtFM)
        fpM = cv2.bitwise_and(resM, gtBM)
        tp = np.sum(tpM)
        fp = np.sum(fpM)
        recall[i] += tp / (gtF + EPS) 
        total = EPS + tp + fp
        precision[i] += (tp + EPS) / total
        tpr[i] += (tp + EPS) / (gtF + EPS)
        fpr[i] += (fp + EPS) / (gtB + EPS)
    np.divide(gtFM, 255.0)
    np.divide(resS, 255.0)
    resS = cv2.absdiff(gtFM, resS)
    mae += np.sum(resS) / (gtFM.shape[0] * gtFM.shape[1])
    print(mae)
    return mae

def get_AUC(resS, gt):
    precision = np.zeros((NUMBER_THRESHOLD, 1))
    recall = np.zeros((NUMBER_THRESHOLD, 1))
    tpr = np.zeros((NUMBER_THRESHOLD, 1))
    fpr = np.zeros((NUMBER_THRESHOLD, 1))
    mea = evaluate_(resS, gt, precision, recall, tpr, fpr)
    print(recall)
    areaROC = 0.
    for i in range(NUMBER_THRESHOLD):
        areaROC += (tpr[i] + tpr[i - 1]) * (fpr[i - 1] - fpr[i]) / 2.0
    print(areaROC)
    return areaROC

if __name__ == "__main__":
    test2_path = "1036.png"
    test1_path = "temp.jpg"
    test1 = cv2.imread(test1_path)
    test2 = cv2.imread(test2_path)
    # gray1 = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)
    np.set_printoptions(threshold=np.inf)
    get_AUC(test1[:,:,0], test2[:,:,0])