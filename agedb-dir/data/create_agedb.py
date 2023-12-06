########################################################################################
# MIT License

# Copyright (c) 2021 Yuzhe Yang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
########################################################################################
import os
import argparse
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, default="./data")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ages, img_paths = [], []

    for filename in tqdm(os.listdir(os.path.join(args.data_path, 'AgeDB'))):
        _, _, age, gender = filename.split('.')[0].split('_')

        ages.append(age)
        img_paths.append(f"AgeDB/{filename}")

    outputs = dict(age=ages, path=img_paths)
    output_dir = os.path.join(args.data_path, "meta")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "agedb.csv")
    df = pd.DataFrame(data=outputs)
    df.to_csv(str(output_path), index=False)


if __name__ == '__main__':
    main()

"""
age,path,split
31,AgeDB/11706_OliviaHussey_31_f.jpg,train
59,AgeDB/11684_MireilleDarc_59_f.jpg,val
44,AgeDB/7955_GilbertRoland_44_m.jpg,train
61,AgeDB/9352_GeorgesMarchal_61_m.jpg,val
28,AgeDB/3888_TomasMilian_28_m.jpg,val
8,AgeDB/16107_DannyGlover_8_m.jpg,test
34,AgeDB/13784_ThelmaRitter_34_f.jpg,train
74,AgeDB/9945_AliMacGraw_74_f.jpg,train
"""