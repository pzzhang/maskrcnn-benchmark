import os
import torch
import torch.utils.data
from PIL import Image
import sys
import base64
import re
if sys.version_info[0] == 2:
    from cStringIO import StringIO
else:
    from io import BytesIO as StringIO
import numpy as np
import json
from maskrcnn_benchmark.structures.tsv_io import TSVFile, generate_lineidx
from maskrcnn_benchmark.structures.io_common import img_from_base64

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from maskrcnn_benchmark.structures.bounding_box import BoxList

import pdb

class VGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, use_difficult=False, transforms=None, version='1600-400-20'):
        self.root=data_dir
        self.image_set=split
        self._version=version
        self.keep_difficult=use_difficult
        self.transforms=transforms
        self._annopath=os.path.join(self.root,'genome',"Annotations","%s.xml")
        self._data_path=os.path.join(self.root,'genome')
        self._img_path=os.path.join(self.root,'vg')
        self.anno=[]

        self.classes = ['__background__']
        self.class_to_ind = {}
        self.class_to_ind[self.classes[0]] = 0
        with open(os.path.join(self._data_path, self._version,'objects_vocab.txt')) as f:
            count = 1
            for object in f.readlines():
                names = [n.lower().strip() for n in object.split(',')]
                self.classes.append(names[0])
                for n in names:
                    self.class_to_ind[n] = count
                count += 1

        self.attributes=['__no_attribute__']
        self.attribute_to_ind={}
        self.attribute_to_ind[self.attributes[0]]=0
        with open(os.path.join(self._data_path, self._version,'attributes_vocab.txt')) as f:
            count=1
            for att in f.readlines():
                names = [n.lower().strip() for n in att.split(',')]
                self.attributes.append(names[0])
                for n in names:
                    self.attribute_to_ind[n] = count
                count += 1

        self.relations = ['__no_relation__']
        self.relation_to_ind = {}
        self.relation_to_ind[self.relations[0]] = 0
        with open(os.path.join(self._data_path, self._version,'relations_vocab.txt')) as f:
            count = 1
            for rel in f.readlines():
                names = [n.lower().strip() for n in rel.split(',')]
                self.relations.append(names[0])
                for n in names:
                    self.relation_to_ind[n] = count
                count += 1

        cache_file=os.path.join(self.root,'vg_%s_%s.tsv'%(self._version, self.image_set))
        if not os.path.exists(cache_file):
            raise ValueError("The file %s does not exists!"%(cache_file))
        idwh_file = os.path.join(self.root,'vg_%s_%s_idwh.npy'%(self._version, self.image_set))
        if not os.path.exists(idwh_file):
            raise ValueError("The file %s does not exists!"%(idwh_file))
        self.tsvfile=TSVFile(cache_file)
        self.idwh = np.load(idwh_file)
        self.length = self.tsvfile.num_rows()
        assert self.length==len(self.idwh), "lenght of the tsv and lenghth of the idwh should be the same!"
        print (self.length)

    def __getitem__(self,index):
        #print ('yes1')
        img,target,_=self.get_groundtruth(index,True)
        #target=target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img,target=self.transforms(img,target)
        #print ('yes4')
        return img,target,index

    def __len__(self):
        return self.length

    def _get_size(self,index):
        return self.idwh[index,1], self.idwh[index,2]

    def get_groundtruth(self,index,call=False):
        #print ('yes2')
        row=self.tsvfile.seek(index)
        anno=json.loads(row[1])
        image_data=row[2]
        base64_data = base64.b64decode(image_data)
        base64_data = StringIO(base64_data)
        img = Image.open(base64_data).convert("RGB")

        anno["boxes"] = torch.tensor(anno["boxes"], dtype=torch.float32)
        anno["labels"] = torch.tensor(anno["labels"])

        anno["attributes"]=torch.tensor(anno["attributes"])
        anno["relations"]=torch.tensor(anno["relations"])
        height,width=anno["im_info"]

        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("attributes",anno["attributes"])
        target.add_field("relations",anno["relations"])
        #print ('yes3')
        if call:
            return img, target, anno
        else:
            return target

    def get_img_info(self,index):
        return {"height": self.idwh[index,2],"width":self.idwh[index,1]}

    def map_class_id_to_class_name(self,class_id):
        return self.classes[class_id]
    def map_attribute_id_to_attribute_name(self,attribute_id):
        return self.attributes[attribute_id]