"""Engineering checks; these synthetic samples are never publication results."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from test_feature_graph import StagedBackbone, loss_args
from models.feature_graph import FeatureGraphRefiner
from utils.total_loss import FeatureGraphLoss
from datasets.wildfire import WildFire
from experiments.evaluation import cm_metrics, corrupt
from experiments.protocol import audit, prepare, read_csv, write_csv
from experiments.paper import bootstrap


class PaperExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_flags_preserve_default_and_self_edges(self):
        x=torch.rand(1,4,16,16)
        model=FeatureGraphRefiner(StagedBackbone(),teacher=None,hidden_dim=16,
                                 heads=4,layers=1,patch_size=4,nonlocal_k=0,spatial=False)
        model.eval()
        out=model(x,return_details=True)
        self.assertTrue(torch.equal(out['edge_index'][0],out['edge_index'][1]))
        torch.testing.assert_close(out['coarse_logits'],out['refined_logits'],rtol=0,atol=0)
        # Removing evidence zeros all its observable paths, not just its label.
        head=model.refiner
        head.use_evidence=False; head.use_predictions=False; head.use_edge_attr=False
        features=model.backbone(x,return_features=True); features['images']=x
        encoded=head.encode_graph(features['outputs'][1],features['outputs'][2],features)
        self.assertEqual(encoded['evidence'].abs().sum(),0)
        self.assertEqual(encoded['attributes'].abs().sum(),0)
        self.assertEqual(encoded['probs'].abs().sum(),0)

    def test_finetune_control_and_each_changed_path_backward(self):
        x=torch.rand(1,4,16,16); y=torch.randint(0,3,(1,16,16))
        for options in [dict(refiner_kind='none'),dict(use_prior=False),dict(zero_init=False),dict(use_evidence=False)]:
            base=StagedBackbone(); teacher=copy.deepcopy(base)
            m=FeatureGraphRefiner(base,teacher=teacher,hidden_dim=16,heads=4,layers=1,patch_size=4,**options).train()
            args=loss_args()
            if options.get('refiner_kind')=='none': args['GRAPH_NODE_WEIGHT']=0.
            args['TEACHER_SELECTION']='all'
            loss=FeatureGraphLoss(args).get_loss(m(x),y,torch.zeros_like(y))[0].mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(m.backbone.final_layer.weight.grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(m.parameter_groups(.1,.02,.1))

    def test_deterministic_whole_frame_eval_and_leakage_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,value in [('train',40),('val',80),('test',120)]:
                (root/f'{name}.txt').write_text(f'{name}_XXX.png\n',encoding='utf-8')
                Image.fromarray(np.full((20,30,3),value,np.uint8)).save(root/f'{name}_rgb.png')
                Image.fromarray(np.full((20,30),value,np.uint8)).save(root/f'{name}_ir.png')
                gt=np.zeros((20,30),np.uint8);gt[:,:3]=255
                Image.fromarray(gt).save(root/f'{name}_gt.png')
            args=dict(ROOTDATASET=str(root),TRAINSET='train.txt',VALIDSET='val.txt',TESTSET='test.txt')
            self.assertTrue(audit(args,root/'audit')['ok'])
            ds=WildFire(str(root),'test.txt',num_classes=3,eval_resize=True,crop_size=(16,16),label_encoding='palette')
            a,b=ds[0],ds[0]
            np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
            self.assertEqual(a[0].shape,(4,16,16));self.assertTrue((a[1][:,0]==2).all())
            (root/'val.txt').write_text('test_XXX.png\n',encoding='utf-8')
            self.assertFalse(audit(args,root/'audit')['ok'])

    def test_metrics_corruption_manifest_and_paired_bootstrap(self):
        scores=cm_metrics(np.array([[5,1,0],[0,2,1],[1,0,3]]),['background','smoke','fire'])
        self.assertAlmostEqual(scores['fire_iou'],3/5)
        self.assertAlmostEqual(scores['pixel_accuracy'],10/13)
        x=torch.ones(1,4,8,8)
        shifted=corrupt(x,'ir_shift_2')
        self.assertTrue((shifted[:,3,:,:2]==0).all());self.assertTrue((shifted[:,:3]==1).all())
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            rows=prepare(root,seeds=(200,),base_epochs=2,graph_epochs=2)
            self.assertEqual(len(rows),len(__import__('experiments.protocol',fromlist=['VARIANTS']).VARIANTS))
            record=dict(sample_id='frame',output='prediction',**{f'cm_{i}_{j}':10 if i==j else 0 for i in range(3) for j in range(3)})
            write_csv(root/'a.csv',[record]);write_csv(root/'b.csv',[dict(record,output='refined')])
            bootstrap(root/'a.csv',root/'b.csv',root/'ci.csv','prediction','refined',draws=100)
            self.assertTrue(all(float(r['delta'])==0 for r in read_csv(root/'ci.csv')))


if __name__=='__main__': unittest.main()
