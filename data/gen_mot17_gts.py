import os.path as osp
import os
import numpy as np
import argparse

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)



def gen_mot17_gts(seq_root, label_root):
    mkdirs(label_root)
    seqs = [s for s in os.listdir(seq_root)]

    tid_curr = 0
    tid_last = -1
    for i, seq in enumerate(seqs):
        print(i, seq)
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, 'img1')
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if mark == 0 or not label == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            # x += w / 2    # maintain xywh format, same as DanceTrack.
            # y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:d} {:d} {:d} {:d} {:f}\n'.format(
                tid_curr, int(x), int(y), int(w), int(h), float(_))
            with open(label_fpath, 'a') as f:
                f.write(label_str)


# Example usage:
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='List folders in a directory and write them to a file.')
    # Add arguments
    parser.add_argument('--data-dir', type=str, default='/data0/DatasetsForSambaMOTR/MOT17/images/train', help='Directory to list folders from')
    parser.add_argument('--label-dir', type=str, default='/data0/DatasetsForSambaMOTR/MOT17/gts/train', help='Split to generate seqmap for')

    # Parse arguments
    args = parser.parse_args()

    # Use the arguments
    gen_mot17_gts(args.data_dir, args.label_dir)
    print('Done')


if __name__ == '__main__':
    main()