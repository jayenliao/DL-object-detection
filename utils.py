import os
import pandas as pd
from xml.dom import minidom
from args import init_arguments
from sklearn.model_selection import train_test_split

def load_xml_single(folder:str, fn:str):
    d_label = {
        'aquarium': 1, 'bottle': 2, 'bowl': 3, 'box': 4, 'bucket': 5,
        'plastic_bag': 6, 'plate': 7, 'styrofoam': 8, 'tire': 9,
        'toilet': 10, 'tub': 11, 'washing_machine': 12, 'water_tower': 13
    }

    fn_type = fn.split('.')[-1]
    fn = fn.replace(fn_type, 'xml') if fn_type != 'xml' else fn
    folder = folder + '/' if folder[-1] != '/' else folder
    mydoc = minidom.parse(folder + fn)
    img_fn = mydoc.getElementsByTagName('filename')[0].childNodes[0].data
    
    objects = [img_fn]
    for obj in mydoc.getElementsByTagName('object'):
        lst = []
        for i in [1, 3, 5, 7]:
            xy = obj.childNodes[3].childNodes[i].childNodes[0].data
            lst.append(str(xy))
        id_label = d_label[obj.childNodes[1].childNodes[0].data]
        lst.append(str(id_label))
        objects.append(','.join(lst))
    
    return ' '.join(objects)

def load_xml(folder:str):
    lst_xml = os.listdir(folder)
    out = []
    for fn in lst_xml:
        out.append(load_xml_single(folder, fn))
    return out

def object_counts(out, rough=True):
    n_objects = []
    for row in out:
        n_objects.append(row.count(' '))
    
    if rough:
        S = pd.Series(n_objects).value_counts()
        n_objects_rough = []
        for n in n_objects:
            if n in S[S < 10].index:
                n_objects_rough.append(999)
            else:
                n_objects_rough.append(n)
        return n_objects_rough
    else:
        return n_objects

def data_splitting(out:list, val_size:float, test_size:float):
    assert 0 <= val_size <= 1
    assert 0 <= test_size <= 1
    n_objects_rough = object_counts(out, True)
    out_tr, out_ = train_test_split(out,  test_size=val_size+test_size, stratify=n_objects_rough)
    n_objects_rough_ = object_counts(out_, True)
    out_va, out_te = train_test_split(out_, test_size=test_size/(val_size+test_size), stratify=n_objects_rough_)
    return out_tr, out_va, out_te

def save_txt(out:list, subset:str):
    with open(subset+'.txt', 'w') as f:
        for row in out:
            f.write(row + '\n')

if __name__ == '__main__':
    args = init_arguments().parse_args()
    out = load_xml(args.FOLDERxml)
    out_tr, out_va, out_te = data_splitting(out, args.val_size, args.test_size)
    save_txt(out_tr, 'train')
    save_txt(out_va, 'val')
    save_txt(out_te, 'test')
