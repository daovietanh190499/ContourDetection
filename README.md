### Huấn luyện với tập Pascal VOC

Cấu trúc thư mục bao gồm
  - Thư mục cnts chứa các ảnh nhãn
  - Thư mục images chứa các ảnh gốc (JPEGImages)
  - Tệp `train.txt` chứa mã của các ảnh huấn luyện, gồm 10581 ảnh
  - Tệp `val.txt` chứa mã của các ảnh kiểm thử, gồm 1449 ảnh

Thay đổi các tham số trong file train.py

```
self.images_path = "/content/drive/MyDrive/convert_dataset/images/"
self.ctns_path = "/content/drive/MyDrive/convert_dataset/ctns/"
self.train_path = "/content/drive/MyDrive/train.txt"
self.val_path = "/content/drive/MyDrive/val.txt"
self.model_save_path = "/content/drive/MyDrive/model_seg/"
self.model_save_name = "cedn_epoch_9.pth"
```

Huấn luyện bằng đối tượng Trainer

```
from CEDN import CEDN
trainer = Trainer(CEDN())
trainer.train()
```
