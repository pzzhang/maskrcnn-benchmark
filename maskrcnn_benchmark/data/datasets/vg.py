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

        self.ids,self.id_to_dir=self._load_image_set_index()
        self.id_to_img_map={k: v for k,v in enumerate(self.ids)}
        cache_file=os.path.join(self.root,'vg_%s_%s.tsv'%(self._version, self.image_set))
        if not os.path.exists(cache_file):
            self._dump_tsv()
        print (len(self.ids))
        self.tsvfile=TSVFile(cache_file)
        print (self.tsvfile.num_rows())

    def __getitem__(self,index):
        #print ('yes1')
        img,target,_=self.get_groundtruth(index,True)
        #target=target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img,target=self.transforms(img,target)
        #print ('yes4')
        return img,target,index

    def __len__(self):
        return len(self.ids)

    def image_path_from_index(self,index):
        img_id=self.ids[index]
        folder=self.id_to_dir[img_id]

        image_path = os.path.join(self._img_path, folder,
                                  str(img_id) + ".jpg")
        return image_path

    def _get_size(self,index):
        
        return Image.open(self.image_path_from_index(index)).size

    def _load_image_set_index(self):
        if self.image_set.endswith('train'):
            training_split_file=os.path.join(self._data_path,'train.txt')
        else:
            training_split_file = os.path.join(self._data_path, 'val.txt')
        with open(training_split_file) as f:
            metadata = f.readlines()
            if self.image_set == "minitrain":
                metadata = metadata[:1000]
            elif self.image_set == "smalltrain":
                metadata = metadata[:20000]
            elif self.image_set == "minival":
                metadata = metadata[:100]
            elif self.image_set == "smallval":
                metadata = metadata[:2000]
        image_index = []
        id_to_dir = {}
        for line in metadata:
            im_file, ann_file = line.split()
            image_id = int(ann_file.split('/')[-1].split('.')[0])
            filename=self._annopath % image_id
            if os.path.exists(filename):
                tree=ET.parse(filename)
                for obj in tree.findall('object'):
                    obj_name = obj.find('name').text.lower().strip()
                    if obj_name in self.class_to_ind:
                        image_index.append(image_id)
                        id_to_dir[image_id] = im_file.split('/')[0]
                        break
        return image_index,id_to_dir

    def _dump_tsv(self):
        cache_file=os.path.join(self.root,'vg_%s_%s.tsv'%(self._version, self.image_set))
        with open(cache_file,'w') as fout:
            for index in range(len(self.ids)):
                img_id = self.ids[index]
                with open(self.image_path_from_index(index), 'rb') as f:
                    image_data=f.read()
                    image_data=base64.b64encode(image_data)
                    base64_data=base64.b64decode(image_data)
                    base64_data=StringIO(base64_data)
                    img=Image.open(base64_data)
                anno=ET.parse(self._annopath % img_id).getroot()
                width,height=self._get_size(index)
                anno=self._preprocess_annotation(anno,width,height)
                fout.write('%d\t%s\t%s\n' % (index,json.dumps(anno),image_data))

    def get_groundtruth(self,index,call=False):
        #print ('yes2')
        cache_file=os.path.join(self.root,'vg_%s_%s.tsv'%(self._version, self.image_set))
        img_id = self.ids[index]
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

    def _preprocess_annotation(self,target,width,height):
        boxes=[]
        gt_classes=[]
        gt_attributes=[]
        difficult_boxes=[]
        TO_REMOVE=0

        obj_dict = {}
        num=0
        for obj in target.iter("object"):
            name=obj.find("name").text.lower().strip()
            if not name in self.class_to_ind:
                continue
            bb=obj.find("bndbox")
            x1 = max(0, float(bb.find('xmin').text))
            y1 = max(0, float(bb.find('ymin').text))
            x2 = min(width - 1, float(bb.find('xmax').text))
            y2 = min(height - 1, float(bb.find('ymax').text))
            # If bboxes are not positive, just give whole image coords (there are a few examples)
            if x2 < x1 or y2 < y1:
                x1 = 0
                y1 = 0
                x2 = width - 1
                y2 = width - 1
            box=[x1,y1,x2,y2]
            bndbox=tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )
            obj_dict[obj.find('object_id').text] = num

            atts=obj.findall('attribute')
            n=0
            for att in atts:
                att=att.text.lower().strip()
                if att in self.attribute_to_ind:
                    if n==0:
                        gt_attributes.append([self.attribute_to_ind[att]])
                    else:
                        gt_attributes[num].append(self.attribute_to_ind[att])
                    n+=1
                if n >=16:
                    break
            while (len(gt_attributes)>num and len(gt_attributes[num])<16):
                gt_attributes[num].append(0)
            if len(gt_attributes)<=num:
                gt_attributes.append([0])
                for i in range(15):
                    gt_attributes[num].append(0)

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            num+=1

        rels=target.findall('relation')
        num_rels=len(rels)
        gt_relations=[]
        for rel in rels:
            pred=rel.find('predicate').text
            if pred:
                pred=pred.lower().strip()
                if pred in self.relation_to_ind:
                    try:
                        triple=[]
                        triple.append(obj_dict[rel.find('subject_id').text])
                        triple.append(self.relation_to_ind[pred])
                        triple.append(obj_dict[rel.find('object_id').text])
                        gt_relations.append(tuple(triple))
                    except:
                        pass

        im_info=tuple(map(float,(height,width)))

        res={
            "boxes":boxes,
            "labels":gt_classes,
            'attributes':gt_attributes,
            'relations':gt_relations,
            "im_info":im_info,
        }
        return res

    def get_img_info(self,index):
        width, height = self._get_size(index)
        im_info=tuple(map(int, (height, width)))
        return {"height": im_info[0],"width":im_info[1]}

    def map_class_id_to_class_name(self,class_id):
        return self.classes[class_id]
    def map_attribute_id_to_attribute_name(self,attribute_id):
        return self.attributes[attribute_id]