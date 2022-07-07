"""
This file organizes raw EEG & eye-tracking data according to BIDS format.

Assumptions:
-   all EEG files (.eeg, .vhdr, .vmrk) 
    are named in a 'Subject-00X-timestamp' format (e.g. Subject-002-[2022.06.12-14.35.46].eeg)
-   eye-tracking files (.edf)
    are named in a 'sub_X format' (e.g. sub_2.edf)
"""


from pathlib import Path
import argparse
import shutil
import re
import os


def data2bids():
    """
    this function organizes raw eeg&eye-tracking data according to BIDS format
    Arguments:
        source_path: Path where raw data is stored; 
        default: current working directory
        type: str
        
        target_path: Path where BIDS-organized data will be saved
        default: current working directory
        type: str
        
        subjects: List of subject numbers divided by comma (e.g. '1,2,3,4,5')
        type: str
        
        task: Name of the task; default: experimental task
        type: str
    """
    #Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=str,
                        help='source data path')
    parser.add_argument('--target-path', type=str,
                        help='target data path')
    parser.add_argument('-s','--subjects', type=str,
                        help='subjects id numbers')
    parser.add_argument('-t','--task', type=str,
                        help='task name')
    #Do the parsing
    args = parser.parse_args()
    #If source path was not specified...
    if args.source_path is None:
        #source path would be a current working directory
        src_path = Path('.')
    else:
        #else source path would be the specified one
        src_path = Path(args.source_path)
    #If target path was not specified...
    if args.target_path is None:
        #target path would be a current working directory
        tgt_path = Path('.')
    else:
        #else target path would be the specified one
        tgt_path = Path(args.target_path)
    #If task name was not specified...
    if args.task is None:
         #default name would be assigned
        task = 'experimental_task'
    else:
        #else specified task name would be assigned
        task = args.task
    subjects = [int(sub) for sub in args.subjects.split(',')]
    for sub in subjects:
        sub_path = tgt_path / Path('sub-{:02d}'.format(sub))
        if sub_path.exists():
            shutil.rmtree(sub_path)
        sub_path.mkdir()
        eye_path = sub_path / Path('eyetracking')
        eye_path.mkdir()
        edf_path = src_path / Path(f'sub_{sub}.edf')
        shutil.copyfile(
            edf_path,
            eye_path / Path('sub-{:02d}_{}_physio.edf'.format(sub, task)))
        eeg_path = sub_path / Path('eeg')
        eeg_path.mkdir()
        for ext in ['.eeg', '.vhdr', '.vmrk']:
            src = list(src_path.glob('Subject-{:03d}-*{}'.format(sub, ext)))[0]
            tgt = eeg_path / Path('sub-{:02d}_{}_eeg{}'.format(sub, task, ext))
            shutil.copyfile(src, tgt)
            if ext != '.eeg':
                with open(tgt, 'r') as fd:
                    content = fd.read()
                content = re.sub(
                    'MarkerFile=.*\n',
                    'MarkerFile=sub-{:02d}_{}_eeg.vmrk\n'.format(sub, task),
                    content)
                content = re.sub(
                    'DataFile=.*\n',
                    'DataFile=sub-{:02d}_{}_eeg.eeg\n'.format(sub, task),
                    content)
                with open(tgt, 'w') as fd:
                    fd.write(content)
