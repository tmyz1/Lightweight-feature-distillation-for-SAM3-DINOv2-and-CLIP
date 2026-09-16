"""
coco 数据集格式：
{
  'info':{
    'description':'COCO 2017 Dataset',
    'url':'http://cocodataset.org',
    'version':'1.0',
    'year':'2017',
    'contributor':'COCO Consortium',
    'date_created':'2017/09/01',
  },
  'licenses':{},
  'images':[{
    'license':'',
    'file_name':'001.png',
    'coco_url':'',
    'height':640,
    'width':640,
    'date_captured':'',
    'flickr_url':,
    'id':'1'
    }.{...}],
  'annotations':[{
    'segmentation':[[]],
    'area':,
    'iscrowd':0,
    'image_id':xxx,# 和images.id是多对一关系
    'bbox':[],
    'category_id':0-80,
    'id':1，# 每个标注对象自己的唯一编号
  },{...}],
  'categories':[{
    'supercategory':'person',
    'id':1,
    'name':'person'
  },{...}]
}
"""
import argparse
import json
import os
#json_file = r"E:\my_data\COCO\annotations_trainval2017\annotations\instances_train2017.json"
def get_all_json(file_path):
    files = os.listdir(file_path)
    jsons = []
    for file in files:
        if file.endswith(".json"):
            new_file_path = os.path.join(file_path, file)
            jsons.append(new_file_path)
    return jsons
def get_coco_json():
    coco = {}
    coco['info'] = {}
    coco['licenses'] = {}
    coco['images'] = []
    coco['annotations'] = []
    coco['categories'] = [{
        'supercategory':'person',
        'id':1,
        'name':'person'
    }]
    return coco
def deal_with_json_image(json_data,coco):
    json_data = json_data['image']
    image_dict = {
        'licenses':None,
        'file_name':json_data['file_name'],
        'coco_url':None,
        'height': json_data['height'],
        'width': json_data['width'],
        'date_captured': None,
        'flickr_url': None,
        'id': len(coco['images']) + 1,
    }
    coco['images'].append(image_dict)

def deal_with_json_annotation(json_data,coco):
    image_id = json_data['image']['image_id']
    json_data = json_data['annotations']
    for anno in json_data:
        anno_dict = {
            'segmentation': None,
            'area': anno['area'],
            'iscrowd': 0,
            'image_id': image_id,
            'bbox': anno['bbox'],
            'category_id': 1,
            'id': anno['id'],
        }
        coco['annotations'].append(anno_dict)


def deal_with_one_json(file_path,coco):
    json_data = json.load(open(file_path))
    deal_with_json_image(json_data,coco)
    deal_with_json_annotation(json_data,coco)
def main(args):
    jsons = get_all_json(args.file_path)
    coco = get_coco_json()
    for j in jsons:
        deal_with_one_json(j,coco)
        print(f'{j} is done')
    output_file_path = os.path.join(args.output_path,args.output_file_name)
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(coco, json_file, ensure_ascii=False)
    print(f'jsons saved to {output_file_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get one json")
    parser.add_argument("--file_path", type=str, default=r"F:\data\SA-1B\test\train\sa_000000")
    parser.add_argument('--output_file_name', type=str, default="instances_train2017.json")
    parser.add_argument('--output_path',type = str,default = r"F:\data\SA-1B\test\train\sa_000000")
    args = parser.parse_args()
    main(args)
