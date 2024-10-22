# generate **.pth
import os
import sys
import torch
sys.path.append('.')

import argparse
parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--dataset', type=str, choices=['refcoco', 'refcoco+','refcocog_google', 'refcocog_umd'], default='refcoco')
args = parser.parse_args()

def main(args):
    dataset = args.dataset
    input_txt_list = os.listdir(f'../ln_data/anns/{dataset}')
    if not os.path.exists(f'../data/{dataset}'):
        os.makedirs(f'../data/{dataset}')
    for input_txt in input_txt_list:
        split = input_txt.split('_')[-1].split('.')[0]
        input_txt = os.path.join('../ln_data/anns', dataset, input_txt)
        res = []
        with open(input_txt, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split() 
                stop = len(line)
                img_name = line[0]
                for i in range(1,len(line)):
                    if (line[i]=='~'):
                        stop=i
                        break
                box_ = [list(map(int,box.split(','))) for box in line[1:stop]]
                box = box_[0][:4]
                seg_id=box_[0][-1]
                
                sent_stop=stop+1
                for i in range(stop+1,len(line)):
                    if line[i]=='~': 
                        des = ''
                        for word in line[sent_stop:i]:
                            des = des + word + ' '
                        sent_stop=i+1
                        des = des.rstrip(' ')
                        res.append((img_name, seg_id, box, des))
                des = ''
                for word in line[sent_stop:len(line)]:
                    des = des + word + ' '
                des = des.rstrip(' ')
                res.append((img_name, seg_id, box, des))
            # print(res)

        imgset_path = '{0}_{1}.pth'.format(dataset, split)
        images = torch.save(res, os.path.join("../data", dataset, imgset_path))
    print(dataset, " done")

if __name__ == "__main__":
    main(args)
