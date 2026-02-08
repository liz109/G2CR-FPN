import os
import argparse
import pandas as pd
from src.data import data_utils


"""
# FIRST STEP: annotation.py to link data paths

$ python annotation.py 
"""

def ensemble_images(path):
    inputs = data_utils.get_files(os.path.join(path, 'input'), 'npy')
    df = pd.DataFrame({'input': inputs})

    names = df['input'].apply(lambda x: os.path.basename(x))
    df['target'] = names.apply(lambda x: os.path.join(path, 'target', x))
    df['file'] = names.apply(lambda x: x[:-4])
    df['subject'] = df['file'].apply(lambda x: x.split('_')[0])
    df['slice'] = df['file'].apply(lambda x: int(x.split('_')[1]))
    df.sort_values(by=['subject', 'slice'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def main(args):
    annotation_dir = os.path.join(args.output, args.data_name)
    os.makedirs(annotation_dir, exist_ok=True)
    print(f"==> Output directory: {annotation_dir}")

    annotation_file = os.path.join(annotation_dir, f'annotation_{args.case}.pkl')
    if os.path.exists(annotation_file):
        raise FileExistsError(f"Annotation file already exists: {annotation_file}")
    df = ensemble_images(args.input)
    df.to_pickle(annotation_file)


    # -------------- Data split --------------
    if args.split_mode == 'subject':
        train_indices = df[~df['subject'].isin(args.test_subject)].index
        test_indices = df[df['subject'].isin(args.test_subject)].index
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")


    # Save index files
    train_indices_file = os.path.join(annotation_dir, f'train_{args.case}.npy')
    test_indices_file = os.path.join(annotation_dir, f'test_{args.case}.npy')
    data_utils.save_npy(train_indices_file, train_indices)
    data_utils.save_npy(test_indices_file, test_indices)

    print(f"Saved: {train_indices_file}")
    print(f"Saved: {test_indices_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate annotation and split indices for AAPM dataset.')
    parser.add_argument('--input', type=str, 
                        default='~/datasets/AAPM/processed/aapm_all_npy_3mm',
                        help='Input data path.')
    parser.add_argument('--output', type=str, 
                        default='./data',
                        help='Project root path.')
    parser.add_argument('--data_name', type=str, 
                        default='aapm_all_npy_3mm', help='Dataset name.')
    parser.add_argument('--case', type=int, 
                        required=True, help='Case number for split.')
    parser.add_argument('--split_mode', type=str,  
                        default='subject')
    parser.add_argument('--test_subject', type=str, nargs='+', 
                        default=['L506'],
                        help='List of subject ID(s) to use as test set.')

    args = parser.parse_args()
    main(args)
    
    


"""
python annotation.py --case 0 --split_mode all_test --test_subject L506 L067 L286
"""