import os
import cv2
from tqdm import tqdm
try:
    import metrics as M
except:
    from .metrics import metrics as M
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

_EPS = 1e-16
_TYPE = np.float64

def get_files(path,name='.pkl'):
    file_lan = []
    for filepath,dirnames,filenames in os.walk(path):
        for filename in filenames:
            if name not in os.path.join(filepath,filename):
                file_lan.append(os.path.join(filepath,filename))
    return file_lan

def once_compute(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE,IoU,mIoU,PrF1):
    gt_path = os.path.join(gt_root, gt_name)
    pred_path = os.path.join(pred_root, gt_name)
    # print(gt_path,pred_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    
    # Skip if files couldn't be read
    if gt is None or pred is None:
        return None
    
    gtsize = gt.shape
    predsize = pred.shape
    if gtsize[0] == predsize[1] and gtsize[1] == predsize[0] and gtsize[0] != gtsize[1]:
        print(pred_path)
    if predsize[0] != gtsize[0] and predsize[1] != gtsize[1]:
        pred = cv2.resize(pred, (gtsize[1], gtsize[0]))
    precisions,recalls = FM.step(pred=pred, gt=gt)
    wfm = WFM.step(pred=pred, gt=gt)
    mae = MAE.step(pred=pred, gt=gt)
    sm = SM.step(pred=pred, gt=gt)
    em = EM.step(pred=pred, gt=gt)
    iou = IoU.step(pred=pred, gt=gt)
    miou = mIoU.step(pred=pred, gt=gt)
    precision, recall, f1 = PrF1.step(pred=pred, gt=gt)
    return {'precisions':precisions,
            'recalls':recalls,
            'wfm':wfm,
            'mae':mae,
            'sm':sm,
            'em':em,
            'iou':iou,
            'miou':miou,
            'precision':precision,
            'recall':recall,
            'f1':f1,
            }

def once_get(gt_root,pred_root,FM,WFM,SM,EM,MAE,IoU,mIoU,PrF1,testdir,i,n_jobs):
    gt_name_list = get_files(pred_root)
    gt_name_list = sorted([x.split('/')[-1] for x in gt_name_list])
    # print(gt_root,pred_root)
    # for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
    results = Parallel(n_jobs=n_jobs)(delayed(once_compute)(gt_root,gt_name,pred_root,FM,WFM,SM,EM,MAE,IoU,mIoU,PrF1) for gt_name in tqdm(gt_name_list, total=len(gt_name_list)))
    precisions,recalls,wfm,sm,em,mae,iou,miou,precision,recall,f1 = [],[],[],[],[],[],[],[],[],[],[]
    for result in results:
        # Skip None results (files that couldn't be read)
        if result is None:
            continue
        precisions.append([result['precisions']])
        recalls.append([result['recalls']])
        wfm.append([result['wfm']])
        mae.append([result['mae']]) 
        sm.append([result['sm']])
        em.append([result['em']])
        iou.append([result['iou']])
        miou.append([result['miou']])
        precision.append([result['precision']])
        recall.append([result['recall']])
        f1.append([result['f1']])
        
    # print(np.array(fm, dtype=_TYPE).shape)
    precisions = np.array(precisions, dtype=_TYPE)
    recalls = np.array(recalls, dtype=_TYPE)
    
    # Handle case where no valid images were found
    if len(precisions) == 0:
        print(f"WARNING: No valid image pairs found for {testdir}")
        return pd.DataFrame()
    
    precision_curve = precisions.mean(axis=0)
    recall_curve = recalls.mean(axis=0)
    fmeasure = 1.3 * precision_curve * recall_curve / (0.3 * precision_curve + recall_curve + _EPS)
    # print(fm.shape)
    wfm = np.mean(np.array(wfm, dtype=_TYPE))
    mae = np.mean(np.array(mae, dtype=_TYPE))
    sm = np.mean(np.array(sm, dtype=_TYPE))
    em = np.mean(np.array(em, dtype=_TYPE), axis=0)
    iou = np.mean(np.array(iou, dtype=_TYPE))
    miou = np.mean(np.array(miou, dtype=_TYPE))
    precision = np.mean(np.array(precision, dtype=_TYPE))
    recall = np.mean(np.array(recall, dtype=_TYPE))
    f1 = np.mean(np.array(f1, dtype=_TYPE))
    onefile = pd.DataFrame()
    results = {'maxFm':fmeasure.max(),
        'wFmeasure':wfm,
        'MAE':mae, 
        'Smeasure':sm, 
        'meanEm':em.mean(),
        'IoU':iou,
        'mIoU':miou,
        'Precision':precision,
        'Recall':recall,
        'F1-Score':f1,
        }
    results = pd.DataFrame.from_dict([results]).T
    onefile = pd.concat([onefile,results])
    print(
        'testdir:', testdir+'##'+str(i), ', ',
        'maxFm:', fmeasure.max().round(3),'; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'Smeasure:', sm.round(3), '; ',
        'meanEm:', em.mean().round(3), '; ',
        'IoU:', iou.round(3), '; ',
        'mIoU:', miou.round(3), '; ',
        'Precision:', precision.round(3), '; ',
        'Recall:', recall.round(3), '; ',
        'F1-Score:', f1.round(3), '; ',
        sep=' '
    )
    # onefile.to_csv(args.testdir+str(i)+".csv")
    # allfile = pd.concat([allfile.T,onefile.T]).T
    return onefile

def soc_metrics(testdir):
    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()
    IoU = M.IoUScore()
    mIoU = M.mIoU()
    PrF1 = M.PrecisionRecallF1()

    # Paths for evaluation
    # Ground truth masks directory
    gt_roots = [
        r'/Vaibhav/shivasish1/PDFNet/DATA/DIS-DATA/DIS-TE/masks',
    ]
    
    # Predictions directory (where Test.py saves predictions)
    # Predictions are saved as: /Vaibhav/shivasish1/PDFNet/results_simulator_data/{testdir}/{dataset_name}/
    cycle_roots = [
        f'/Vaibhav/shivasish1/PDFNet/results_simulator_data/{testdir}/DIS-VD',
    ]
    
    # Output CSV directory
    output_dir = f'/Vaibhav/shivasish1/PDFNet/results_simulator_data/{testdir}'
    os.makedirs(output_dir, exist_ok=True)
    
    n_jobs = 12

    allfile = pd.DataFrame()
    for i in range(gt_roots.__len__()):
        gt_root = gt_roots[i]
        pred_root = cycle_roots[i]
        onefile = once_get(gt_root,pred_root,FM,WFM,SM,EM,MAE,IoU,mIoU,PrF1,testdir,i,n_jobs)
        if not onefile.empty:
            allfile = pd.concat([allfile.T,onefile.T]).T
    
    # Save results to CSV
    if not allfile.empty:
        allfile.to_csv(os.path.join(output_dir, "ALL.csv"))
        print(f"\nResults saved to: {os.path.join(output_dir, 'ALL.csv')}")
    else:
        print(f"\nWARNING: No valid results to save. Check if predictions exist in: {cycle_roots}")



if __name__ == '__main__':
    # soc_metrics('BiRefNet-UH')
    # soc_metrics('Inspy-UH')
    # soc_metrics('PGNet-UH')
    soc_metrics('HRSOD_VIDP-d2-p8-loss_ablation-baseline2025-03-13_14_40_48')