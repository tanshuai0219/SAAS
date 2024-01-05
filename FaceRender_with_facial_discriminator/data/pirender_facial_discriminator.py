import os
import lmdb
import random
import collections
import numpy as np
from PIL import Image
from io import BytesIO
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import pickle

def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')

class PIrenderDataset(Dataset):
    def __init__(self, opt, is_inference):
        path = opt.path
        self.env = lmdb.open(
            os.path.join(path),
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


        self.mead_path = opt.mead_path
        self.hdtf_path = opt.hdtf_path
        self.eye_enlarge_ratio = opt.eye_enlarge_ratio
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)
        list_file = "test_facial.json" if is_inference else "train_facial.json"
        list_file = os.path.join(path, "lists",list_file)

        with open(list_file,"r") as f:
            videos = json.load(f)
        self.resolution = opt.resolution
        self.semantic_radius = opt.semantic_radius
        self.video_items, self.person_ids, self.person_id_meads, self.person_ids_emotion = self.get_video_index(videos)
        self.idx_by_person_id = self.group_by_key(self.video_items, key='person_id')
        self.idx_by_person_id_emotion = self.group_by_key(self.video_items, key='person_id_emotion')
        self.idx_by_person_id_mead = self.group_by_key(self.video_items, key='person_id_mead')
        self.person_ids = self.person_ids #* 100

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ])

    def get_component_coordinates(self, name, index):
        """Get facial component (left_eye, right_eye, mouth) coordinates from a pre-loaded pth file"""
        # components_bbox = self.components_list[name]
        splits = name.split('#')
        if len(splits)==2:
            components_bbox = torch.load(os.path.join(self.hdtf_path, 'eye_mouth_landmarks', name+'.npy'))
        else:
            a,b,c,d = splits
            components_bbox = torch.load(os.path.join(self.mead_path, a, 'eye_mouth_landmarks',b,c, d+'.npy'))
        with self.env.begin(write=False) as txn:
            key = format_for_lmdb(name, 'length')
            length = int(txn.get(key).decode('utf-8'))
        # get coordinates
        locations = []
        for part in ['left_eye', 'right_eye', 'mouth']:
            mean = components_bbox[part][index][0:2]
            half_len = components_bbox[part][index][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations.append(loc)
        return locations

    def get_video_index(self, videos):
        video_items = []
        person_ids = []
        person_id_meads = []
        person_ids_emotion = []
        tot_len = len(videos)
        print('loading video_index')
        pbar = tqdm(range(tot_len))
        for i in pbar:
            video  = videos[i]
            video_items.append(self.Video_Item(video))
            splits = video.split('#')
            if len(splits) == 2:
                person_ids.append(splits[0])
                person_id_meads.append('lll')
                person_ids_emotion.append(splits[0]+'#'+splits[1])
            else:
                a,b,c,d = splits
                person_ids.append( a+'#'+b+'#'+c)
                person_id_meads.append(a)
                person_ids_emotion.append(splits[0]+'#'+splits[1])

        person_ids = sorted(person_ids)
        person_id_meads = sorted(person_id_meads)
        person_ids_emotion = sorted(person_ids_emotion)

        return video_items, person_ids, person_id_meads, person_ids_emotion

    def group_by_key(self, video_list, key):
        return_dict = collections.defaultdict(list)
        for index, video_item in enumerate(video_list):
            return_dict[video_item[key]].append(index)
        return return_dict  
    
    def Video_Item(self, video_name):
        video_item = {}
        splits = video_name.split('#')
        if len(splits) == 2:
            video_item['video_name'] = video_name
            video_item['person_id'] = video_name.split('#')[0] # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = 'lll'
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item
        else:
            a,b,c,d = splits
            video_item['video_name'] = video_name
            video_item['person_id'] = a+'#'+b+'#'+c # M003#angry#030 WDA_DonnaShalala1_000#29
            video_item['person_id_mead'] = a
            video_item['person_id_emotion'] = video_name.split('#')[0]+'#'+video_name.split('#')[1]
            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], 'length')
                length = int(txn.get(key).decode('utf-8'))
            video_item['num_frame'] = length
            
            return video_item

    def __len__(self):
        return len(self.person_ids)

    def __getitem__(self, index):
        data={}
        person_id = self.person_ids[index]
        
        if len(person_id.split('#'))==3:
            person_id_neutral = person_id.split('#')[0]+'#neutral'
            source_video_item = self.video_items[random.choices(self.idx_by_person_id_emotion[person_id_neutral], k=1)[0]]
            
            target_video_item = self.video_items[random.choices(self.idx_by_person_id_mead[person_id.split('#')[0]], k=1)[0]]

            frame_source, frame_target = self.random_select_frames_source_target(source_video_item,target_video_item)

            source_locations = self.get_component_coordinates(source_video_item['video_name'], frame_source)
            source_loc_left_eye, source_loc_right_eye, source_loc_mouth = source_locations


            target_locations = self.get_component_coordinates(target_video_item['video_name'], frame_target)
            target_loc_left_eye, target_loc_right_eye, target_loc_mouth = target_locations

            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(source_video_item['video_name'], frame_source) # M027#neutral#014-0000153
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(target_video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)
                # print(video_item['video_name'], frame_source, frame_target)
                source_semantics_key = format_for_lmdb(source_video_item['video_name'], 'coeff_3dmm') # M027#neutral#014-0000153
                source_semantics_numpy = np.frombuffer(txn.get(source_semantics_key), dtype=np.float32)

                source_semantics_numpy = source_semantics_numpy.reshape((-1,260)) # video_item['num_frame']

                if source_semantics_numpy.shape[0]<source_video_item['num_frame']:
                    source_semantics_numpy = np.concatenate((source_semantics_numpy,source_semantics_numpy[::-1]), axis=0)
                if source_semantics_numpy.shape[0]> source_video_item['num_frame']:
                    source_semantics_numpy = source_semantics_numpy[:source_video_item['num_frame']]

                # print(video_item['video_name'], frame_source, frame_target)
                target_semantics_key = format_for_lmdb(target_video_item['video_name'], 'coeff_3dmm')
                target_semantics_numpy = np.frombuffer(txn.get(target_semantics_key), dtype=np.float32)

                target_semantics_numpy = target_semantics_numpy.reshape((-1,260)) # video_item['num_frame']

                if target_semantics_numpy.shape[0]<target_video_item['num_frame']:
                    target_semantics_numpy = np.concatenate((target_semantics_numpy,target_semantics_numpy[::-1]), axis=0)
                if target_semantics_numpy.shape[0]> target_video_item['num_frame']:
                    target_semantics_numpy = target_semantics_numpy[:target_video_item['num_frame']]

            img1 = Image.open(BytesIO(img_bytes_1))
            data['source_image'] = self.transform(img1)

            img2 = Image.open(BytesIO(img_bytes_2))
            data['target_image'] = self.transform(img2) 

            data['target_semantics'] = self.transform_semantic(target_semantics_numpy, frame_target)
            data['source_semantics'] = self.transform_semantic(source_semantics_numpy, frame_source)

            data['target_loc_left_eye'] = target_loc_left_eye
            data['target_loc_right_eye'] = target_loc_right_eye
            data['target_loc_mouth'] = target_loc_mouth

            data['source_loc_left_eye'] = source_loc_left_eye
            data['source_loc_right_eye'] = source_loc_right_eye
            data['source_loc_mouth'] = source_loc_mouth

            return data
        else:
            video_item = self.video_items[random.choices(self.idx_by_person_id[person_id], k=1)[0]]
            
            frame_source, frame_target = self.random_select_frames(video_item)

            source_locations = self.get_component_coordinates(video_item['video_name'], frame_source)
            source_loc_left_eye, source_loc_right_eye, source_loc_mouth = source_locations


            target_locations = self.get_component_coordinates(video_item['video_name'], frame_target)
            target_loc_left_eye, target_loc_right_eye, target_loc_mouth = target_locations


            with self.env.begin(write=False) as txn:
                key = format_for_lmdb(video_item['video_name'], frame_source)
                img_bytes_1 = txn.get(key) 
                key = format_for_lmdb(video_item['video_name'], frame_target)
                img_bytes_2 = txn.get(key)
                # print(video_item['video_name'], frame_source, frame_target)
                semantics_key = format_for_lmdb(video_item['video_name'], 'coeff_3dmm')
                semantics_numpy = np.frombuffer(txn.get(semantics_key), dtype=np.float32)
                # print(semantics_numpy.shape[0])
                # print(int(semantics_numpy.shape[0]/video_item['num_frame']))
                # semantics_numpy = semantics_numpy[:int(semantics_numpy.shape[0]/video_item['num_frame'])*257]
                # print(semantics_numpy.shape[0])
                # if semantics_numpy.shape[0]/260
                semantics_numpy = semantics_numpy.reshape((-1,260)) # video_item['num_frame']
                # semantics_numpy = semantics_numpy[:video_item['num_frame']]
                if semantics_numpy.shape[0]<video_item['num_frame']:
                    semantics_numpy = np.concatenate((semantics_numpy,semantics_numpy[::-1]), axis=0)
                if semantics_numpy.shape[0]> video_item['num_frame']:
                    semantics_numpy = semantics_numpy[:video_item['num_frame']]
                # print(semantics_numpy.shape)

            img1 = Image.open(BytesIO(img_bytes_1))
            data['source_image'] = self.transform(img1)

            img2 = Image.open(BytesIO(img_bytes_2))
            data['target_image'] = self.transform(img2) 

            data['target_semantics'] = self.transform_semantic(semantics_numpy, frame_target)
            data['source_semantics'] = self.transform_semantic(semantics_numpy, frame_source)
            
            
            data['target_loc_left_eye'] = target_loc_left_eye
            data['target_loc_right_eye'] = target_loc_right_eye
            data['target_loc_mouth'] = target_loc_mouth

            data['source_loc_left_eye'] = source_loc_left_eye
            data['source_loc_right_eye'] = source_loc_right_eye
            data['source_loc_mouth'] = source_loc_mouth
            return data
    
    def random_select_frames(self, video_item):
        if video_item['num_frame']<=10:
            num_frame = video_item['num_frame']
        else:
            num_frame = video_item['num_frame']-10
        frame_idx = random.choices(list(range(num_frame)), k=2)
        return frame_idx[0], frame_idx[1]

    def random_select_frames_source_target(self, source_video_item, target_video_item):
        if source_video_item['num_frame']<=10:
            num_frame = source_video_item['num_frame']
        else:
            num_frame = source_video_item['num_frame']-10
        
        from_frame_idx = random.choices(list(range(num_frame)), k=1)
        if target_video_item['num_frame']<=10:
            num_frame_target = target_video_item['num_frame']
        else:
            num_frame_target = target_video_item['num_frame']-10
        target_frame_idx = random.choices(list(range(num_frame_target)), k=1)
        return from_frame_idx[0], target_frame_idx[0]

    def transform_semantic(self, semantic, frame_index):
        index = self.obtain_seq_index(frame_index, semantic.shape[0])
        coeff_3dmm = semantic[index,...]
        # id_coeff = coeff_3dmm[:,:80] #identity
        ex_coeff = coeff_3dmm[:,80:144] #expression
        # tex_coeff = coeff_3dmm[:,144:224] #texture
        angles = coeff_3dmm[:,224:227] #euler angles for pose
        # gamma = coeff_3dmm[:,227:254] #lighting
        translation = coeff_3dmm[:,254:257] #translation
        crop = coeff_3dmm[:,257:260] #crop param

        coeff_3dmm = np.concatenate([ex_coeff, angles, translation, crop], 1)
        return torch.Tensor(coeff_3dmm).permute(1,0)

    def obtain_seq_index(self, index, num_frames):
        seq = list(range(index-self.semantic_radius, index+self.semantic_radius+1))
        seq = [ min(max(item, 0), num_frames-1) for item in seq ]
        return seq



