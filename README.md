# DL-object-detection

## Data

Please go to the page of the competition on the AIdea [here](https://aidea-web.tw/topic/cc2d8ec6-dfaf-42bd-8a4a-435bffc8d071) to download the following data.

- `./train_cdc/train_annotations/`: 提供與 train_images 資料夾下檔名對應之標註資料XML檔，此 XML 檔仿照 PASCAL VOC 格式，內容提供標註影像的檔名(filename)、影像大小(size)及標註物件(object)。每個標註物件(object)會提供其物件名稱(name)、標註 bounding box 之左上座標(xmin, ymin)與右下座標(xmax, ymax)
- `./train_cdc/train_images/`: 訓練所需的影像資料（JPG格式），共 2671 張。
- `./test_cdc/test_images/`：測試所需的影像資料（JPG格式），共 2248 張(包含 test_pub_cdc.zip 的資料)。

## Codes

- `args.py` defines the argument parser.
- `utils.py`: tools for data loading 

## Usage
