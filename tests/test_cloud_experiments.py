"""CPU checks for GPU assignment and the separate timing output contract."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from experiments.cloud import jobs_for,run_jobs
from experiments.evaluation import benchmark
from experiments.protocol import prepare,read_csv


class CloudExperimentTests(unittest.TestCase):
    def test_prepare_uses_portable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'suite'
            rows=prepare(root,seeds=[200],base_epochs=2,graph_epochs=2)
            for row in rows:
                self.assertNotIn('\\',row['checkpoint'])
            self.assertEqual(len(rows),22)

    def test_independent_gpu_assignment_and_log_creation(self):
        rows=[dict(experiment='egr_fs',seed=str(s),dependency='rffnet',family='main') for s in (200,201,202)]
        launched=[]
        class Process:
            pid=123
            def __init__(self,command,**kwargs): launched.append((command,kwargs['env']))
            def poll(self): return 0
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);jobs=jobs_for(root,rows,'val')
            with patch('experiments.cloud.done',return_value=False),patch('experiments.cloud.subprocess.Popen',Process),patch('experiments.cloud.time.sleep'):
                run_jobs(root,jobs,['0','1'],False,root/'queue.csv')
            self.assertEqual([env['CUDA_VISIBLE_DEVICES'] for _,env in launched],['0','1','0'])
            self.assertTrue(all(cmd[-1]=='cuda:0' for cmd,_ in launched))
            self.assertTrue(all(row['status']=='completed' for row in read_csv(root/'queue.csv')))
            self.assertEqual(len(list((root/'queue_logs').glob('*.log'))),3)

    def test_benchmark_has_no_accuracy_results(self):
        model=torch.nn.Conv2d(4,3,1)
        class Data:
            files=['one_reference_image']
            def __getitem__(self,index): return (np.zeros((4,8,8),np.float32),)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'weights.pth'
            torch.save(dict(model_state_dict=model.state_dict()),checkpoint)
            with patch('experiments.evaluation.get_model',return_value=model),patch('experiments.evaluation.get_dataset',return_value=(None,Data(),None)):
                benchmark(dict(SEED=200,EXPERIMENT_ID='engineering'),checkpoint,root/'timing',device='cpu',warmup=1,repeats=2)
            self.assertFalse((root/'timing'/'summary.csv').exists())
            self.assertTrue((root/'timing'/'completed.json').exists())
            self.assertEqual(len(read_csv(root/'timing'/'latency_samples.csv')),2)


if __name__=='__main__': unittest.main()
